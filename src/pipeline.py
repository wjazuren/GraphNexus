import json
import traceback
from typing import Literal
from models import *
from utils import *
from modules import *
from construct import *


class Pipeline:
    def __init__(self, llm: BaseEngine):
        self.llm = llm
        # 案例仓库
        self.case_repo = CaseRepositoryHandler(llm = llm)
        # 模型智能体
        self.schema_agent = SchemaAgent(llm = llm)

        self.extraction_agent = ExtractionAgent(llm = llm, case_repo = self.case_repo)
        self.reflection_agent = ReflectionAgent(llm = llm, case_repo = self.case_repo)

    def __check_consistancy(self, llm, task, mode, update_case):
        if llm.name == "OneKE":
            if task == "Base" or task == "Triple":
                raise ValueError("The finetuned OneKE only supports quick extraction mode for NER, RE and EE Task.")
            else:
                mode = "quick"
                update_case = False
                print("The fine-tuned OneKE defaults to quick extraction mode without case update.")
                return mode, update_case
        return mode, update_case

    # 默认执行顺序固定：schema_agent → extraction_agent → reflection_agent
    def __init_method(self, data: DataPoint, method):
        default_order = ["schema_agent", "extraction_agent", "reflection_agent"]
        
        # Ensure method is a dictionary
        if not isinstance(method, dict):
            method = {"extraction_agent": "extract_information_direct"}
        
        if "schema_agent" not in method:
            method["schema_agent"] = "get_default_schema"
        if data.task != "Base":
            method["schema_agent"] = "get_retrieved_schema"
        if "extraction_agent" not in method:
            method["extraction_agent"] = "extract_information_direct"
        sorted_process_method = {key: method[key] for key in default_order if key in method}
        return sorted_process_method
    '''
    当外部没有完整传入指令、schema 时，自动加载全局配置内置 prompt 与输出结构：
    NER：实体抽取
    RE：关系抽取
    EE：事件抽取
    Triple：三元组抽取
    '''
    def __init_data(self, data: DataPoint):
        if data.task == "NER":
            data.instruction = config['agent']['default_ner']
            data.output_schema = "EntityList"
        elif data.task == "RE":
            data.instruction = config['agent']['default_re']
            data.output_schema = "RelationList"
        elif data.task == "EE":
            data.instruction = config['agent']['default_ee']
            data.output_schema = "EventList"
        elif data.task == "Triple":
            data.instruction = config['agent']['default_triple']
            data.output_schema = "TripleList"
        return data

    # main entry
    def get_extract_result(self,
                           task: TaskType,
                           three_agents = {},
                           construct = {},
                           instruction: str = "",
                           text: str = "",
                           output_schema: str = "",
                           constraint: str = "",
                           use_file: bool = False,
                           file_path: str = "",
                           truth: str = "",
                           mode: str = "quick",
                           update_case: bool = False,
                           show_trajectory: bool = False,
                           isgui: bool = False, 
                           iskg: bool = False,
                           config_name: str = "", 
                           ):

        logger.info(f"\n========Pipeline Start ========")
        logger.info(f"Input task: {task}, mode: {mode}, isgui: {isgui}, config_name: {config_name}")

        # Check Consistancy
        # 一致性检测
        mode, update_case = self.__check_consistancy(self.llm, task, mode, update_case)

        # Load Data
        data = DataPoint(task=task, instruction=instruction, text=text, output_schema=output_schema, constraint=constraint, use_file=use_file, file_path=file_path, truth=truth)
        data = self.__init_data(data)
        # 读取配置文档
        if mode in config['agent']['mode'].keys():
            process_method = config['agent']['mode'][mode].copy()
        else:
            process_method = mode
        # 使用前端传入的 three_agents 自定义流程
        if isgui and mode == "customized":
            process_method = three_agents
            logger.info("Customized 3-Agents: %s", three_agents)

        sorted_process_method = self.__init_method(data, process_method)
        logger.info(f"Process Method: {sorted_process_method}")

        print_schema = False 
        frontend_schema = "" 
        frontend_res = "" 

        # Information Extract
        # 动态反射调用：通过字符串找到对应智能体和方法，扩展性极强。
        # 执行顺序固定：SchemaAgent → ExtractionAgent → ReflectionAgent。
        for agent_name, method_name in sorted_process_method.items():
            logger.info(f"--> [Executing Agent]: {agent_name} -> {method_name}")
            agent = getattr(self, agent_name, None)
            if not agent:
                logger.error(f"Agent '{agent_name}' not found in Pipeline.")
                continue
            method = getattr(agent, method_name, None)
            if not method:
                logger.error(f"Method '{method_name}' not found in Agent '{agent_name}'.")
                continue
            
            try:
                data = method(data)
                logger.info(f" {agent_name}.{method_name} executed successfully.")
            except Exception as e:
                logger.error(f" Failed during {agent_name}.{method_name}: {e}")
                traceback.print_exc()

            if not print_schema and getattr(data, 'print_schema', None): 
                logger.info(f"Schema Generated: {data.print_schema}\n")
                frontend_schema = data.print_schema
                print_schema = True

        # Summarize Answer
        # 汇总答案
        logger.info(f"\nSummarizing answers from Extraction Agent...")
        if self.extraction_agent is not None:
            try:
                data = self.extraction_agent.summarize_answer(data)
                # logger.info(f" raw_response in data -> {getattr(data, 'response', 'N/A')}")
                logger.info(f"Final Pred: data.pred -> {data.pred}")
            except Exception as e:
                logger.error(f" Error during summarize_answer: {e}")
                traceback.print_exc()
        else:
            data.pred = []

        # Force format prediction result to JSON string
        # 结果序列化 + 自动保存 JSON
        extraction_result = ""
        try:
            if isinstance(data.pred, (dict, list)):
                extraction_result = json.dumps(data.pred, indent=4, ensure_ascii=False)
            else:
                extraction_result = str(data.pred)
        except Exception as e:
            logger.error(f" Failed to serialize data.pred to JSON: {e}")
            extraction_result = str(data.pred)

        logger.info(f"\n======== 📊 [Final Output Result] ========\n{extraction_result[:200]}\n===============================================\n")

        # Save result to file
        if config_name:
            try:
                import os
                result_dir = "examples/results"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                
                base_name = os.path.splitext(os.path.basename(config_name))[0]
                result_file_path = os.path.join(result_dir, f"{base_name}.json")
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    f.write(extraction_result)
                logger.info(f"💾 [SUCCESS]: Extraction Result saved to: {result_file_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save result file: {e}")

        # Construct KG
        # 写入知识图谱
        if iskg:
            try:
                myurl = construct['url']
                myusername = construct['username']
                mypassword = construct['password']
                logger.info(f"Construct KG in your {construct['database']} now...")
                source = data.file_path if data.use_file and data.file_path else (config_name or "inline")
                source_text = "\n".join(data.chunk_text_list)
                ingestion_stats = upsert_structured_facts(
                    uri=myurl,
                    user=myusername,
                    password=mypassword,
                    extraction_result=extraction_result,
                    source=source,
                    source_text=source_text,
                    database=construct.get('database', 'neo4j'),
                )
                logger.info(f"Structured ingestion completed: {ingestion_stats}")
            except Exception as e:
                logger.error(f"❌ Failed during KG construction: {e}")

        frontend_res = data.pred

        # Case Update
        # 更新案例库
        if update_case:
            if self.case_repo is None:
                logger.warning("Warning: Case update not available - CaseRepositoryHandler not loaded.")
            else:
                if (data.truth == ""):
                    truth = input("Please enter the correct answer you prefer, or just press Enter to accept the current answer: ")
                    if truth.strip() == "":
                        data.truth = data.pred
                    else:
                        data.truth = extract_json_dict(truth)
                self.case_repo.update_case(data)

        # Return result
        result = data.pred
        trajectory = data.get_result_trajectory()

        return result, trajectory, frontend_schema, frontend_res
