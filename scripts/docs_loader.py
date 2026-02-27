# docs_loader.py
import os
import zipfile
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Prefer PyMuPDF (fitz) si está instalado; fallback se informa
try:
    import fitz
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import docx
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


def extract_zip(zip_path: str, out_dir: str) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip no encontrado: {zip_path}")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    logger.info("Zip extraído a %s", out_dir)


def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    texts = []
    if _HAS_FITZ:
        try:
            doc = fitz.open(path)
            for i in range(doc.page_count):
                page = doc.load_page(i)
                txt = page.get_text("text")
                texts.append((i + 1, txt or ""))
            return texts
        except Exception as e:
            logger.warning("fitz fallo en %s: %s", path, str(e))
    # Si no hay PyMuPDF devolvemos el archivo como single page vacío y avisamos
    logger.warning("PyMuPDF no disponible o fallo en %s. Instala 'PyMuPDF' para mejor extracción.", path)
    return [(1, "")]


def extract_text_from_docx(path: str) -> List[Tuple[int, str]]:
    if not _HAS_DOCX:
        logger.warning("python-docx no disponible. Instalad 'python-docx' para soporte .docx")
        return [(1, "")]
    try:
        doc = docx.Document(path)
        full = []
        for p in doc.paragraphs:
            if p.text:
                full.append(p.text)
        return [(1, "\n".join(full))]
    except Exception as e:
        logger.exception("Error leyendo docx %s: %s", path, e)
        return [(1, "")]


def extract_text_from_xlsx(path: str) -> List[Tuple[int, str]]:
    if not _HAS_PANDAS:
        logger.warning("pandas no disponible. Instalad 'pandas' para soporte xlsx/csv.")
        return [(1, "")]
    try:
        xls = pd.ExcelFile(path)
        texts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, header=None, dtype=str)
            df = df.fillna("")
            text = "\n".join([" ".join(row.astype(str).tolist()) for _, row in df.iterrows()])
            texts.append(f"--- Hoja: {sheet} ---\n" + text)
        return [(1, "\n\n".join(texts))]
    except Exception as e:
        logger.exception("Error leyendo xlsx %s: %s", path, e)
        return [(1, "")]


def extract_text_from_csv(path: str) -> List[Tuple[int, str]]:
    if not _HAS_PANDAS:
        logger.warning("pandas no disponible. Instalad 'pandas' para soporte xlsx/csv.")
        return [(1, "")]
    try:
        df = pd.read_csv(path, dtype=str)
        df = df.fillna("")
        text = "\n".join([" ".join(row.astype(str).tolist()) for _, row in df.iterrows()])
        return [(1, text)]
    except Exception as e:
        logger.exception("Error leyendo csv %s: %s", path, e)
        return [(1, "")]


def extract_text_from_txt(path: str) -> List[Tuple[int, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return [(1, txt)]
    except Exception as e:
        logger.exception("Error leyendo txt %s: %s", path, e)
        return [(1, "")]


def _smart_paragraph_split(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r'\n{1,}', text) if p.strip()]
    if not paras:
        return [text[i:i+500] for i in range(0, len(text), 500)]
    return paras


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    paragraphs = _smart_paragraph_split(text)
    chunks = []
    cur = ""
    for p in paragraphs:
        if not p:
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end])
                    start = end - overlap if end - overlap > start else end
                cur = ""
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def load_documents_from_folder(folder_path: str, max_chars: int = 1000, overlap: int = 200) -> List[Dict]:
    entries = []
    p = Path(folder_path)
    if not p.exists():
        logger.warning("Carpeta no existe: %s", folder_path)
        return entries

    for fname in sorted(os.listdir(folder_path)):
        path = p / fname
        if path.is_dir():
            continue
        lower = fname.lower()
        try:
            if lower.endswith(".zip"):
                tmp_dir = p / f"_extracted_{fname}"
                tmp_dir.mkdir(exist_ok=True)
                extract_zip(str(path), str(tmp_dir))
                entries += load_documents_from_folder(str(tmp_dir), max_chars=max_chars, overlap=overlap)
                continue
            elif lower.endswith(".pdf"):
                page_texts = extract_text_from_pdf(str(path))
            elif lower.endswith(".docx"):
                page_texts = extract_text_from_docx(str(path))
            elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                page_texts = extract_text_from_xlsx(str(path))
            elif lower.endswith(".csv"):
                page_texts = extract_text_from_csv(str(path))
            elif lower.endswith(".txt"):
                page_texts = extract_text_from_txt(str(path))
            else:
                logger.debug("Formato no soportado (skip): %s", fname)
                continue

            for page_no, page_text in page_texts:
                if not page_text or not page_text.strip():
                    continue
                chunks = chunk_text(page_text, max_chars=max_chars, overlap=overlap)
                for i, ch in enumerate(chunks):
                    entry_id = f"{fname}::p{page_no}::chunk_{i}"
                    entries.append({
                        "id": entry_id,
                        "text": ch,
                        "meta": {"source": fname, "page": page_no, "chunk_index": i}
                    })
            logger.info("Procesado %s -> %d chunks", fname, len([e for e in entries if e["meta"]["source"] == fname]))
        except Exception as e:
            logger.exception("Error cargando %s: %s", fname, e)
    logger.info("Total chunks cargados: %d", len(entries))
    return entries


def save_chunks_to_jsonl(chunks: List[Dict], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for c in chunks:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Chunks guardados en %s", out_path)


def load_chunks_from_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    result = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            result.append(json.loads(line))
    logger.info("Cargados %d chunks desde %s", len(result), path)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cargar documentos y generar chunks (docs_loader).")
    parser.add_argument("--input", "-i", required=True, help="Carpeta con documentos o zip.")
    parser.add_argument("--out", "-o", default="export/chunks.jsonl", help="Ruta de salida JSONL.")
    parser.add_argument("--max_chars", type=int, default=1000, help="Tamaño máximo por chunk en caracteres.")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap en caracteres entre chunks.")
    args = parser.parse_args()

    docs = load_documents_from_folder(args.input, max_chars=args.max_chars, overlap=args.overlap)
    # tras docs = load_documents_from_folder(DOCS_FOLDER)
    st.write("DEBUG: total chunks indexados:", len(docs))
# mostrar primeras 50 ids para inspección
    for i, entry in enumerate(docs[:50]):
    # docs puede ser (id,text) o dict; normalizar para evitar errores
        try:
            sid = entry[0]
        except Exception:
            sid = str(entry)
        st.write(f"{i+1}. {sid}")
    save_chunks_to_jsonl(docs, args.out)
    logger.info("Finalizado. %d chunks exportados.", len(docs))