from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

import pymupdf

fitz = pymupdf


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()
    return text


def normalize_pdf_text(page_text: str) -> str:
    cleaned = clean_text(page_text)
    lines = [line.strip() for line in cleaned.split("\n")]
    filtered = []
    for line in lines:
        if not line:
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def extract_pdf_text(file_path: Path) -> List[Dict[str, Any]]:
    sections = []
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = normalize_pdf_text(page.get_text())
        if text:
            sections.append({
                "source": file_path.name,
                "page": page_number + 1,
                "text": text,
            })
    doc.close()
    return sections


def extract_txt_text(file_path: Path) -> List[Dict[str, Any]]:
    text = clean_text(file_path.read_text(encoding="utf-8"))
    if not text:
        return []
    return [{
        "source": file_path.name,
        "page": "N/A",
        "text": text,
    }]


def load_documents(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            documents.extend(extract_pdf_text(file_path))
        elif file_path.suffix.lower() == ".txt":
            documents.extend(extract_txt_text(file_path))
    return documents
