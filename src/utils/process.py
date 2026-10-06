"""
Data Processing Functions.
Supports:
- Segmentation of long text
- Segmentation of file content
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
from nltk.tokenize import sent_tokenize
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
import re
import json
import yaml
import os
import yaml
import os
import inspect
import ast
import pdfplumber
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from utils.logger import logger
from dotenv import load_dotenv

load_dotenv()
with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as file:
    config = yaml.safe_load(file)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# python -X utf8 get_pdf.py
# Load configuration
def load_extraction_config(yaml_path):
    # Read YAML content from the file path
    if not os.path.exists(yaml_path):
        print(f"Error: The config file '{yaml_path}' does not exist.")
        return {}

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract the 'extraction' configuration dictionary
    model_config = config.get('model', {})
    extraction_config = config.get('extraction', {})

    # Model config
    model_name_or_path = model_config.get('model_name_or_path', "")
    model_category = model_config.get('category', "")
    api_key = model_config.get('api_key', "") or os.getenv("LLM_API_KEY", "")
    base_url = model_config.get('base_url', "")
    vllm_serve = model_config.get('vllm_serve', False)

    # Extraction config
    task = extraction_config.get('task', "")
    instruction = extraction_config.get('instruction', "")
    text = extraction_config.get('text', "")
    output_schema = extraction_config.get('output_schema', "")
    constraint = extraction_config.get('constraint', "")
    truth = extraction_config.get('truth', "")
    use_file = extraction_config.get('use_file', False)
    file_path = extraction_config.get('file_path', "")
    mode = extraction_config.get('mode', "quick")
    update_case = extraction_config.get('update_case', False)
    show_trajectory = extraction_config.get('show_trajectory', False)

    # Construct config (optional: for constructing your knowledge graph)
    if 'construct' in config:
        construct_config = config.get('construct', {})
        database = construct_config.get('database', "")
        url = construct_config.get('url', "")
        username = construct_config.get('username', "")
        password = construct_config.get('password', "") or os.getenv("NEO4J_PASSWORD", "")
        # Return a dictionary containing these variables
        return {
            "model": {
                "model_name_or_path": model_name_or_path,
                "category": model_category,
                "api_key": api_key,
                "base_url": base_url,
                "vllm_serve": vllm_serve
            },
            "extraction": {
                "task": task,
                "instruction": instruction,
                "text": text,
                "output_schema": output_schema,
                "constraint": constraint,
                "truth": truth,
                "use_file": use_file,
                "file_path": file_path,
                "mode": mode,
                "update_case": update_case,
                "show_trajectory": show_trajectory
            },
            "construct": {
                "database": database,
                "url": url,
                "username": username,
                "password": password
            }
        }

    # Return a dictionary containing these variables
    return {
        "model": {
            "model_name_or_path": model_name_or_path,
            "category": model_category,
            "api_key": api_key,
            "base_url": base_url,
            "vllm_serve": vllm_serve
        },
        "extraction": {
            "task": task,
            "instruction": instruction,
            "text": text,
            "output_schema": output_schema,
            "constraint": constraint,
            "truth": truth,
            "use_file": use_file,
            "file_path": file_path,
            "mode": mode,
            "update_case": update_case,
            "show_trajectory": show_trajectory
        }
    }

# Split the string text into chunks
def chunk_str(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if current_length + token_count <= config['agent']['chunk_token_limit']:
            current_chunk.append(sentence)
            current_length += token_count
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = token_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def is_complex_pdf(file_path: str) -> bool:
    """
    判断 PDF 是否为需要重型模型（Docling）处理的“复杂/异常文档”
    判定依据：
    1. 含有“表格”类关键字，但 pdfplumber 抽取不到标准线框表格。
    2. 含有乱码、无法识别的非标准 CMap 乱码字符。
    3. 文本行发生异常重叠/重复。
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                
                # 特征 1：页面明显提示有表格，但 pdfplumber 提取结果为空（说明是无框/背景色表格）
                has_table_keyword = any(kw in text for kw in ["表格", "表 1", "表1", "表 2", "表2"])
                extracted_tables = page.extract_tables()
                if has_table_keyword and not extracted_tables:
                    return True
                
                # 特征 2：检测到 CMap 字体编码异常字符或零宽占位符
                if len(re.findall(r'[\u200b-\u200f\ufeff\ufffd]', text)) > 2:
                    return True

                # 特征 3：页面提取出的文本存在乱码或重复行模式
                if len(text) > 200 and text.count("设备状态") > 4:
                    return True

    except Exception:
        # 如果 pdfplumber 直接抛出异常，说明 PDF 结构损坏或极其特殊，强制走 Docling
        return True

    return False


def load_simple_pdf(file_path: str) -> list[Document]:
    """标准 PDF 极速解析器（使用 pdfplumber，毫秒级完成）"""
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                for t in tables:
                    html_table = table_to_html(t)
                    docs.append(
                        Document(
                            page_content=f"【表格数据】:\n{html_table}",
                            metadata={"page": page_num, "type": "table", "parser": "pdfplumber"}
                        )
                    )
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"page": page_num, "type": "text", "parser": "pdfplumber"}
                    )
                )
    return docs


def table_to_html(table: list[list[str]]) -> str:
    """辅助函数：将二维数组转换为 HTML 表格"""
    if not table:
        return ""
    html = ["<table border='1'>"]
    for row_idx, row in enumerate(table):
        html.append("  <tr>")
        for cell in row:
            cell_text = (cell or "").replace("\n", " ").strip()
            tag = "th" if row_idx == 0 else "td"
            html.append(f"    <{tag}>{cell_text}</{tag}>")
        html.append("  </tr>")
    html.append("</table>")
    return "\n".join(html)


def load_complex_pdf(file_path: str) -> list[Document]:
    """复杂 PDF 深度解析器（使用 Docling 重构版面与表格）"""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()

    return [
        Document(
            page_content=markdown_content,
            metadata={"source": file_path, "parser": "docling"}
        )
    ]


def load_pdf_with_table(file_path: str) -> list[Document]:
    """
    智能路由 PDF 加载入口：
    先检测 PDF 复杂度，再决定调用 pdfplumber 还是 Docling
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指定的 PDF 文件: {file_path}")

    # 1. 自动判定文档复杂度
    if is_complex_pdf(file_path):
        logger.info(f"检测到复杂/非标 PDF [{os.path.basename(file_path)}]，启用 Docling 深度重构策略...")
        return load_complex_pdf(file_path)
    else:
        logger.info(f"检测到标准 PDF [{os.path.basename(file_path)}]，启用 pdfplumber 极速解析策略...")
        return load_simple_pdf(file_path)


def chunk_file(file_path: str):
    """统一文件处理入口"""
    pages = []

    if file_path.endswith(".pdf"):
        # 根据智能路由处理 PDF
        pages = load_pdf_with_table(file_path)
    else:
        # 其他文本格式处理
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".html"):
            loader = BSHTMLLoader(file_path, open_encoding='utf-8', bs_kwargs={'features': 'lxml'})
        elif file_path.endswith(".json"):
            loader = JSONLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
            
        pages = loader.load_and_split()

    # 高效拼接所有文本块
    docs = "".join([item.page_content for item in pages])
    logger.info(f"文档解析完成，总字符数: {len(docs)}")
    
    # 进行二次 chunk 重新切片
    pages = chunk_str(docs)

    return pages
# 将单引号替换为双引号
def process_single_quotes(text):
    result = re.sub(r"(?<!\w)'|'(?!\w)", '"', text)
    return result
# 消除多余空格
def remove_empty_values(data):
    def is_empty(value):
        return value is None or value == [] or value == "" or value == {}
    if isinstance(data, dict):
        return {
            k: remove_empty_values(v)
            for k, v in data.items()
            if not is_empty(v)
        }
    elif isinstance(data, list):
        return [
            remove_empty_values(item)
            for item in data
            if not is_empty(item)
        ]
    else:
        return data
'''
场景：LLM 经常输出自然语言 + 夹杂 JSON
使用正则提取文本中最后一个大括号 JSON 块
处理 unicode 转义
json.loads 解析
调用remove_empty_values清理空字段
解析失败则返回原始字符串
'''
def extract_json_dict(text):
    if isinstance(text, dict):
        return text
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    matches = re.findall(pattern, text)
    if matches:
        json_string = matches[-1]
        json_string = json_string.encode('utf-8').decode('unicode_escape')
        # json_string = process_single_quotes(json_string)
        try:
            json_dict = json.loads(json_string)
            json_dict = remove_empty_values(json_dict)
            if json_dict is None:
                return "No valid information found."
            return json_dict
        except json.JSONDecodeError:
            return json_string
    else:
        return text

def good_case_wrapper(example: str):
    if example is None or example == "":
        return ""
    example = f"\nHere are some examples:\n{example}\n(END OF EXAMPLES)\nRefer to the reasoning steps and analysis in the examples to help complete the extraction task below.\n\n"
    return example

def bad_case_wrapper(example: str):
    if example is None or example == "":
        return ""
    example = f"\nHere are some examples of bad cases:\n{example}\n(END OF EXAMPLES)\nRefer to the reflection rules and reflection steps in the examples to help optimize the original result below.\n\n"
    return example

def example_wrapper(example: str):
    if example is None or example == "":
        return ""
    example = f"\nHere are some examples:\n{example}\n(END OF EXAMPLES)\n\n"
    return example
# 消除多余连续空格
def remove_redundant_space(s):
    s = ' '.join(s.split())
    s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)
    return s

def format_string(s):
    s = remove_redundant_space(s)
    s = s.lower()
    s = s.replace('{','').replace('}','')
    s = re.sub(',+', ',', s)
    s = re.sub(r'\.+', '.', s)
    s = re.sub(';+', ';', s)
    s = s.replace('’', "'")
    return s
# 计算评价指标
def calculate_metrics(y_truth: set, y_pred: set):
    TP = len(y_truth & y_pred)
    FN = len(y_truth - y_pred)
    FP = len(y_pred - y_truth)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score
# 调用当前函数的上层函数名称

def current_function_name():
    try:
        stack = inspect.stack()
        if len(stack) > 1:
            outer_func_name = stack[1].function
            return outer_func_name
        else:
            print("No caller function found")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        pass

def normalize_obj(value):
    if isinstance(value, dict):
        return frozenset((k, normalize_obj(v)) for k, v in value.items())
    elif isinstance(value, (list, set, tuple)):
        return tuple(Counter(map(normalize_obj, value)).items())
    elif isinstance(value, str):
        return format_string(value)
    return value

def dict_list_to_set(data_list):
    result_set = set()
    try:
        for dictionary in data_list:
            value_tuple = tuple(format_string(value) for value in dictionary.values())
            result_set.add(value_tuple)
        return result_set
    except Exception as e:
        print (f"Failed to convert dictionary list to set: {data_list}")
        return result_set
