"""文档解析器 — 支持 PDF / Word / TXT / Markdown"""
import re
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunk:
    """文档分块数据结构"""
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


class DocumentParser:
    """多格式文档解析 + 智能分块"""

    @staticmethod
    def parse(file_path: Path) -> str:
        """根据文件类型调用对应解析器"""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return DocumentParser._parse_docx(file_path)
        elif ext in (".txt", ".md"):
            return DocumentParser._parse_text(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    @staticmethod
    def chunk(text: str, metadata: dict = None) -> List[DocumentChunk]:
        """智能分块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [
            DocumentChunk(
                content=chunk,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    # ── 内部解析器 ──

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        doc = pymupdf.open(path)
        texts = []
        for page in doc:
            # 优先提取文本
            text = page.get_text()
            if text.strip():
                texts.append(text)
            # 尝试提取表格
            tables = page.find_tables()
            for table in tables:
                texts.append(DocumentParser._table_to_markdown(table))
        doc.close()
        return "\n\n".join(texts)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        doc = DocxDocument(path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                # 保留标题层级
                if para.style.name.startswith("Heading"):
                    level = para.style.name.split()[-1]
                    prefix = "#" * int(level) if level.isdigit() else "#"
                    paragraphs.append(f"{prefix} {para.text}")
                else:
                    paragraphs.append(para.text)
        # 表格
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                paragraphs.append("\n".join(rows))
        return "\n\n".join(paragraphs)

    @staticmethod
    def _parse_text(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _table_to_markdown(table) -> str:
        """PyMuPDF 表格 → Markdown 表格"""
        rows = table.extract()
        if not rows:
            return ""
        md = []
        for i, row in enumerate(rows):
            md.append("| " + " | ".join(str(c) if c else "" for c in row) + " |")
            if i == 0:
                md.append("|" + "|".join(["---"] * len(row)) + "|")
        return "\n".join(md)
