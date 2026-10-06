from models import *
from utils import *
from .knowledge_base import schema_repository
from langchain_core.output_parsers import JsonOutputParser
from utils.logger import logger

class SchemaAnalyzer:
    def __init__(self, llm: BaseEngine):
        self.llm = llm

    def serialize_schema(self, schema) -> str:
        if isinstance(schema, (str, list, dict, set, tuple)):
            return schema
        try:
            parser = JsonOutputParser(pydantic_object = schema)
            schema_description = parser.get_format_instructions()
            schema_content = re.findall(r'```(.*?)```', schema_description, re.DOTALL)
            explanation = "For example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}}, the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance."
            schema = f"{schema_content}\n\n{explanation}"
        except:
            return schema
        return schema
    # 生成一句话描述文本所属领域、体裁；
    def redefine_text(self, text_analysis):
        try:
            field = text_analysis['field']
            genre = text_analysis['genre']
        except:
            return text_analysis
        prompt = f"This text is from the field of {field} and represents the genre of {genre}."
        return prompt

    '''
    仅在自动推导模式 get_deduced_schema 启用：
    让 LLM 分析原文：属于什么领域（科技 / 小说 / 医疗等）、文本体裁；产出结构化 json，再转换成领域描述 prompt。
    '''
    def get_text_analysis(self, text: str):
        output_schema = self.serialize_schema(schema_repository.TextDescription)
        prompt = text_analysis_instruction.format(examples="", text=text, schema=output_schema)
        response = self.llm.get_chat_response(prompt)
        response = extract_json_dict(response)
        response = self.redefine_text(response)
        return response

    # 直接让模型输出 JSON Schema，返回（code 占位，schema=json 结构体）
    def get_deduced_schema_json(self, instruction: str, text: str, distilled_text: str):
        prompt = deduced_schema_json_instruction.format(examples=example_wrapper(json_schema_examples), instruction=instruction, distilled_text=distilled_text, text=text)
        response = self.llm.get_chat_response(prompt)
        response = extract_json_dict(response)
        code = response
        logger.info(f"Deduced Schema in Json: \n{response}\n\n")
        return code, response
    # generate schema code
    def get_deduced_schema_code(self, instruction: str, text: str, distilled_text: str):
        prompt = deduced_schema_code_instruction.format(examples=example_wrapper(code_schema_examples), instruction=instruction, distilled_text=distilled_text, text=text)
        response = self.llm.get_chat_response(prompt)
        code_blocks = re.findall(r'```[^\n]*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            try:
                code_block = code_blocks[-1]
                namespace = {}
                exec(code_block, namespace)
                schema = namespace.get('ExtractionTarget')
                if schema is not None:
                    index = code_block.find("class")
                    code = code_block[index:]
                    logger.info(f"Deduced Schema in Code: \n{code}\n\n")
                    schema = self.serialize_schema(schema)
                    return code, schema
            except Exception as e:
                logger.warning(f"get_deduced_schema_code解析失败:{e},降级调用get_deduced_schema_json")
                return self.get_deduced_schema_json(instruction, text, distilled_text)
        return self.get_deduced_schema_json(instruction, text, distilled_text)

class SchemaAgent:
    def __init__(self, llm: BaseEngine):
        self.llm = llm
        self.module = SchemaAnalyzer(llm = llm)
        self.schema_repo = schema_repository
        self.methods = ["get_default_schema", "get_retrieved_schema", "get_deduced_schema"]

    # 文本分块：文件读取/直接字符串分块
    def __preprocess_text(self, data: DataPoint):
        if data.use_file:
            data.chunk_text_list = chunk_file(data.file_path)
        else:
            data.chunk_text_list = chunk_str(data.text)
        if data.task == "NER":
            data.print_schema = """
class Entity(BaseModel):
    name : str = Field(description="The specific name of the entity. ")
    type : str = Field(description="The type or category that the entity belongs to.")
class EntityList(BaseModel):
    entity_list : List[Entity] = Field(description="Named entities appearing in the text.")
            """
        elif data.task == "RE":
            data.print_schema = """
class Relation(BaseModel):
    head : str = Field(description="The starting entity in the relationship.")
    tail : str = Field(description="The ending entity in the relationship.")
    relation : str = Field(description="The predicate that defines the relationship between the two entities.")

class RelationList(BaseModel):
    relation_list : List[Relation] = Field(description="The collection of relationships between various entities.")
            """
        elif data.task == "EE":
            data.print_schema = """
class Event(BaseModel):
    event_type : str = Field(description="The type of the event.")
    event_trigger : str = Field(description="A specific word or phrase that indicates the occurrence of the event.")
    event_argument : dict = Field(description="The arguments or participants involved in the event.")

class EventList(BaseModel):
    event_list : List[Event] = Field(description="The events presented in the text.")
            """
        elif data.task == "Triple":
            data.print_schema = """
class Triple(BaseModel):
    head: str = Field(description="The subject or head of the triple.")
    head_type: str = Field(description="The type of the subject entity.")
    relation: str = Field(description="The predicate or relation between the entities.")
    relation_type: str = Field(description="The type of the relation.")
    tail: str = Field(description="The object or tail of the triple.")
    tail_type: str = Field(description="The type of the object entity.")
class TripleList(BaseModel):
    triple_list: List[Triple] = Field(description="The collection of triples and their types presented in the text.")
"""
        return data
    '''
    执行文本分块预处理
    读取全局配置 config['agent']['default_schema']
    将默认 schema 存入 DataPoint
    更新推理轨迹 trajectory
    适用场景：无预定义 schema 仓库、简单任务兜底方案。
    '''
    def get_default_schema(self, data: DataPoint):
        data = self.__preprocess_text(data)
        default_schema = config['agent']['default_schema']
        data.set_schema(default_schema)
        function_name = current_function_name()
        data.update_trajectory(function_name, default_schema)
        return data

    def get_retrieved_schema(self, data: DataPoint):
        self.__preprocess_text(data)
        schema_name = data.output_schema
        schema_class = getattr(self.schema_repo, schema_name, None)
        if schema_class is not None:
            schema = self.module.serialize_schema(schema_class)
            default_schema = config['agent']['default_schema']
            data.set_schema(f"{default_schema}\n{schema}")
            function_name = current_function_name()
            data.update_trajectory(function_name, schema)
        else:
            logger.warning(f"未找到指定 schema: {schema_name},降级为get_default_schema")
            return self.get_default_schema(data)
        return data
    '''
    完整链路：
    文本分块
    取第一段文本调用 module.get_text_analysis() → 判断领域、体裁
    生成 distilled_text（领域描述 prompt）
    调用 get_deduced_schema_code()，尝试让 LLM 写出 Pydantic 抽取模型
    将生成的 Python 代码存入data.print_schema
    将序列化后的 schema 约束存入data.schema
    适用场景：Base 通用抽取任务，事先不知道实体 / 关系结构，需要模型自主定义输出格式。
    '''
    def get_deduced_schema(self, data: DataPoint):
        self.__preprocess_text(data)
        target_text = data.chunk_text_list[0]
        analysed_text = self.module.get_text_analysis(target_text)
        if len(data.chunk_text_list) > 1:
            prefix = "Below is a portion of the text to be extracted. "
            analysed_text = f"{prefix}\n{target_text}"
        distilled_text = self.module.redefine_text(analysed_text)
        code, deduced_schema = self.module.get_deduced_schema_code(data.instruction, target_text, distilled_text)
        data.print_schema = code
        data.set_distilled_text(distilled_text)
        default_schema = config['agent']['default_schema']
        data.set_schema(f"{default_schema}\n{deduced_schema}")
        function_name = current_function_name()
        data.update_trajectory(function_name, deduced_schema)
        return data