"""文档解析 — PDF/Word/TXT/表格提取"""
from pathlib import Path
from typing import List

import pymupdf
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def parse_document(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    if ext in (".docx", ".doc"):
        return _parse_docx(file_path)
    if ext in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")

    raise ValueError(f"不支持的文件类型: {ext}")


def chunk_text(text: str, meta: dict = None) -> List[dict]:
    """把长文本切成适合检索的小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""],
    )
    chunks = splitter.split_text(text)

    base_meta = meta or {}
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "content": chunk,
            "metadata": {**base_meta, "chunk_index": i, "chunk_total": len(chunks)},
        })
    return result


def _parse_pdf(path: Path) -> str:
    doc = pymupdf.open(path)
    parts = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            parts.append(text)

        # 表格提取
        tables = page.find_tables()
        for tbl in tables:
            rows = tbl.extract()
            if rows:
                md_rows = []
                for ri, row in enumerate(rows):
                    cells = [str(c) if c else "" for c in row]
                    md_rows.append("| " + " | ".join(cells) + " |")
                    if ri == 0:
                        md_rows.append("|" + "|".join(["---"] * len(row)) + "|")
                parts.append("\n".join(md_rows))

    doc.close()
    return "\n\n".join(parts)


def _parse_docx(path: Path) -> str:
    doc = DocxDocument(path)
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # 标题层级转 markdown
        style = para.style.name
        if style.startswith("Heading"):
            level = style.split()[-1]
            prefix = "#" * int(level) if level.isdigit() else "#"
            parts.append(f"{prefix} {text}")
        else:
            parts.append(text)

    # 提取表格
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            parts.append("\n".join(rows))

    return "\n\n".join(parts)
