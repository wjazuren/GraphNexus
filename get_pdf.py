import os
import sys

# 1. 彻底解决 Windows GBK 编码的核心设置：强制 Python 采用 UTF-8 编码模式
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
# python -X utf8 get_pdf.py
# 2. 彻底禁用 PyTorch 触发的 JIT 编译编码 BUG
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 3. 设置国内镜像源与抑制警告
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

logging.getLogger("docling").setLevel(logging.ERROR)

def load_pdf_with_docling(file_path: str) -> list[Document]:
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()
    
    return [
        Document(
            page_content=markdown_content,
            metadata={"source": file_path, "parser": "docling"}
        )
    ]

if __name__ == "__main__":
    file_path = r"D:\lianbao\1.pdf"
    
    docs = load_pdf_with_docling(file_path)
    print("\n🎉 解析完全成功！Markdown 结构如下：\n" + "="*50)
    print(docs[0].page_content)