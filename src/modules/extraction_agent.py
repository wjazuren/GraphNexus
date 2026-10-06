from models import *
from utils import *
from .knowledge_base.case_repository import CaseRepositoryHandler
from utils.logger import logger
class InformationExtractor:
    def __init__(self, llm: BaseEngine):
        self.llm = llm
    '''
    对召回的案例做包装 good_case_wrapper；无案例时为空字符串；
    组装完整 Prompt：任务指令 + Few-shot 案例 + 待抽取文本 + 约束条件 + Schema 格式要求；
    请求 LLM，打印模型原生输出；
    第一层清洗：剥离json Markdown 代码块，只保留内部文本；
    第一层解析：优先标准json.loads()；失败降级使用项目工具函数extract_json_dict；
    特殊兜底修复（专门解决 Triple 任务常见 bug）
    痛点：模型有时错误返回 Schema 定义，而非真实triple_list数组。
    判定：response["triple_list"]是字典（schema 模板）而非列表
    动作：正则直接从原始字符串抓取[ {...} ]数组，强行重建{"triple_list": [...]}
    '''
    def extract_information(self, instruction="", text="", examples="", schema="", additional_info=""):
        examples = good_case_wrapper(examples)
        prompt = extract_instruction.format(
            instruction=instruction, 
            examples=examples, 
            text=text, 
            additional_info=additional_info, 
            schema=schema
        )
        
        # 1. 获取大模型原始返回
        raw_response = self.llm.get_chat_response(prompt)
        logger.info(f"\n LLM 原生输出:\n{raw_response}\n")

        # 2. 强力清理 Markdown 和多余空白
        clean_response = raw_response
        if "```" in clean_response:
            # 匹配 ```json ... ``` 内部的核心 JSON 内容
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_response, re.DOTALL)
            if json_match:
                clean_response = json_match.group(1).strip()
            else:
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()

        # 3. 尝试解析 JSON，如果失败则回溯原提取函数
        try:
            response = json.loads(clean_response)
        except Exception as e:
            logger.warning(f"标准 json.loads 解析失败，降级尝试 extract_json_dict: {e}")
            response = extract_json_dict(clean_response)

        # 4. 如果解析出的数据依然是那个无用的 Schema 定义，则尝试强制强行抓取 [ ... ] 列表内容
        if isinstance(response, dict) and "triple_list" in response and isinstance(response["triple_list"], dict):
            
            array_match = re.search(r'\[\s*\{.*\}\s*\]', raw_response, re.DOTALL)
            if array_match:
                try:
                    parsed_array = json.loads(array_match.group(0))
                    response = {"triple_list": parsed_array}
                except Exception as ex:
                    logger.error("❌ [DEBUG Exception] 强制提取列表失败:", ex)

        return response
    '''
    兼容分支，专门给 OneKE 模型 使用，独立 prompt 模板 extract_instruction_json。
    '''
    def extract_information_compatible(self, task="", text="", constraint=""):
        instruction = instruction_mapper.get(task)
        prompt = extract_instruction_json.format(instruction=instruction, constraint=constraint, input=text)
        response = self.llm.get_chat_response(prompt)
        response = extract_json_dict(response)
        return response
    '''
    接收多个文本块各自抽取得到的result_list，调用 LLM 做结果融合、去重、统一整理。
    多个 chunk 分段抽取 → 多个结果 → 汇总为一份标准输出，存入data.pred
    '''
    def summarize_answer(self, instruction="", answer_list="", schema="", additional_info=""):
        prompt = summarize_instruction.format(instruction=instruction, answer_list=answer_list, schema=schema, additional_info=additional_info)
        response = self.llm.get_chat_response(prompt)
        response = extract_json_dict(response)
        return response

class ExtractionAgent:
    def __init__(self, llm: BaseEngine, case_repo: CaseRepositoryHandler):
        self.llm = llm
        self.module = InformationExtractor(llm = llm)
        self.case_repo = case_repo
        self.methods = ["extract_information_direct", "extract_information_with_case"]

    # 遍历所有任务类型（NER/RE/EE/Triple），格式化约束文本，追加进 Prompt
    def __get_constraint(self, data: DataPoint):
        if data.constraint in ("", [], {}, None):
            return data
        if data.task == "NER":
            constraint = json.dumps(data.constraint)
            if "**Entity Type Constraint**" in constraint or self.llm.name == "OneKE":
                return data
            data.constraint = f"\n**Entity Type Constraint**: The type of entities must be chosen from the following list.\n{constraint}\n"
        elif data.task == "RE":
            constraint = json.dumps(data.constraint)
            if "**Relation Type Constraint**" in constraint or self.llm.name == "OneKE":
                return data
            data.constraint = f"\n**Relation Type Constraint**: The type of relations must be chosen from the following list.\n{constraint}\n"
        elif data.task == "EE":
            constraint = json.dumps(data.constraint)
            if "**Event Extraction Constraint**" in constraint:
                return data
            if self.llm.name != "OneKE":
                data.constraint = f"\n**Event Extraction Constraint**: The event type must be selected from the following dictionary keys, and its event arguments should be chosen from its corresponding dictionary values. \n{constraint}\n"
            else:
                try:
                    result = [
                                {
                                    "event_type": key,
                                    "trigger": True,
                                    "arguments": value
                                }
                                for key, value in data.constraint.items()
                            ]
                    data.constraint = json.dumps(result)
                except:
                    print("Invalid Constraint: Event Extraction constraint must be a dictionary with event types as keys and lists of arguments as values.", data.constraint)
        elif data.task == "Triple":
            constraint = json.dumps(data.constraint)
            if "**Triple Extraction Constraint**" in constraint:
                return data
            if self.llm.name != "OneKE":
                print(f"data.constraint为{len(data.constraint)}")
                if len(data.constraint) == 1: # 1 list means entity
                    data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{constraint}\n"
                elif len(data.constraint) == 2: # 2 list means entity and relation
                    if data.constraint[0] == []:
                        data.constraint = f"\n**Triple Extraction Constraint**: Relation type must chosen from following list:\n{data.constraint[1]}\n"
                    elif data.constraint[1] == []:
                        data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{data.constraint[0]}\n"
                    else:
                        data.constraint = f"\n**Triple Extraction Constraint**: Entities type must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\n"
                elif len(data.constraint) == 3: # 3 list means entity, relation and object
                    if data.constraint[0] == []:
                        data.constraint = f"\n**Triple Extraction Constraint**: Relation type must chosen from following list:\n{data.constraint[1]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                    elif data.constraint[1] == []:
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                    elif data.constraint[2] == []:
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\n"
                    else:
                        data.constraint = f"\n**Triple Extraction Constraint**: Subject Entities must chosen from following list:\n{data.constraint[0]}\nRelation type must chosen from following list:\n{data.constraint[1]}\nObject Entities must chosen from following list:\n{data.constraint[2]}\n"
                else:
                    data.constraint = f"\n**Triple Extraction Constraint**: The type of entities must be chosen from the following list:\n{constraint}\n"
            else:
                print("OneKE does not support Triple Extraction task now, please wait for the next version.")
            # print("data.constraint", data.constraint)
        return data

    # 不调用案例库，examples 为空；
    # 文本会被chunk_str切分成多段，逐段调用 LLM 抽取；
    # 每一段抽取结果存入result_list；
    # 执行完成后返回 data，交给流水线执行 summarize_answer。
    # 如果你想要开启案例增强，切换方法为：extract_information_with_case
    def extract_information_direct(self, data: DataPoint):
        data = self.__get_constraint(data)
        result_list = []
        for chunk_text in data.chunk_text_list:
            if self.llm.name != "OneKE":
                extract_direct_result = self.module.extract_information(instruction=data.instruction, text=chunk_text, schema=data.output_schema, examples="", additional_info=data.constraint)
            else:
                extract_direct_result = self.module.extract_information_compatible(task=data.task, text=chunk_text, constraint=data.constraint)
            result_list.append(extract_direct_result)
        function_name = current_function_name()
        data.set_result_list(result_list)
        # 添加到轨迹中(执行日志)
        data.update_trajectory(function_name, result_list)
        return data

    def extract_information_with_case(self, data: DataPoint):
        data = self.__get_constraint(data)
        result_list = []
        for chunk_text in data.chunk_text_list:
            examples = self.case_repo.query_good_case(data)
            extract_case_result = self.module.extract_information(instruction=data.instruction, text=chunk_text, schema=data.output_schema, examples=examples, additional_info=data.constraint)
            result_list.append(extract_case_result)
        function_name = current_function_name()
        data.set_result_list(result_list)
        data.update_trajectory(function_name, result_list)
        return data

    def summarize_answer(self, data: DataPoint):
        if len(data.result_list) == 0:
            return data
        if len(data.result_list) == 1:
            data.set_pred(data.result_list[0])
            return data
        summarized_result = self.module.summarize_answer(instruction=data.instruction, answer_list=data.result_list, schema=data.output_schema, additional_info=data.constraint)
        funtion_name = current_function_name()
        data.set_pred(summarized_result)
        data.update_trajectory(funtion_name, summarized_result)
        return data