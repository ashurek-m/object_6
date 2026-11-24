"""ЭТАП 1 RAG — индексация документов (запускается при обновлении документов).

Пайплайн индексации:
    файлы (data/docs) -> текст -> нарезка на куски -> эмбеддинги -> векторная БД Chroma

Почему куски, а не целые документы:
    контекстное окно LLM ограничено, а при поиске важна точность:
    короткий кусок «Дефектная ведомость оформляется в течение 3 дней»
    ищется лучше, чем документ на 40 страниц.

Запуск:  python ingest.py   (или run_ingest.bat)
"""

import re
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from docx import Document as DocxDocument

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DB_DIR,
    DOCS_DIR,
)
from llm_client import embed

# Сколько текстов отправлять в /embeddings за один запрос (батч — ради скорости)
EMBED_BATCH = 32

# Какие форматы документов понимаем
SUPPORTED = {".md", ".txt", ".docx"}


# --------------------------------------------------------------------------
# 1. Загрузка документов
# --------------------------------------------------------------------------

def _load_docx(path: Path) -> str:
    """Вытащить текст из Word-документа.

    Заголовки стилей Heading 1/2/3 превращаем в markdown-заголовки (#, ##),
    чтобы дальше работало определение разделов. Таблицы выгружаем построчно
    с разделителем «|» — в регламентах в таблицах много важного.
    """
    doc = DocxDocument(str(path))
    lines: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        match = re.match(r"heading (\d)", style)
        if match:
            lines.append("#" * int(match.group(1)) + " " + text)
        else:
            lines.append(text)
            lines.append("")  # пустая строка = граница абзаца

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                lines.append(line)
        lines.append("")
    return "\n".join(lines)


def load_documents() -> list[dict]:
    """Прочитать все поддерживаемые файлы из папки документов.

    Возвращает список словарей: {"source": имя файла, "text": текст}.
    """
    docs: list[dict] = []
    if not DOCS_DIR.exists():
        return docs
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED or path.is_dir():
            continue
        if path.suffix.lower() == ".docx":
            text = _load_docx(path)
        else:  # .md и .txt читаем как есть; utf-8-sig — чтобы не тащил BOM из Windows-блокнота
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


# --------------------------------------------------------------------------
# 2. Чанкинг — нарезка текста на куски
# --------------------------------------------------------------------------

def _tail(text: str, limit: int) -> str:
    """Последние ~limit символов текста, не начиная с обрубленного слова.

    Из этого «хвоста» формируется перекрытие со следующим куском:
    если ответ находится ровно на границе кусков, он попадёт в оба.
    """
    tail = text[-limit:].strip()
    return tail.split(" ", 1)[1].strip() if " " in tail else tail


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Разрезать один огромный абзац по словам с перекрытием."""
    words = paragraph.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > CHUNK_SIZE:
            pieces.append(current)
            current = _tail(current, CHUNK_OVERLAP)
        current = f"{current} {word}" if current else word
    if current.strip():
        pieces.append(current)
    return pieces


def split_into_chunks(text: str, source: str) -> list[dict]:
    """Нарезать документ на куски ~CHUNK_SIZE символов с перекрытием.

    Логика:
      1. Делим текст на абзацы (по пустым строкам) — абзац не рвём без нужды.
      2. Мелкие абзацы склеиваем, пока не наберётся CHUNK_SIZE.
      3. Абзац больше лимита режем отдельно (по словам, с перекрытием).
      4. Между соседними кусками оставляем перекрытие CHUNK_OVERLAP.

    У каждого куска — метаданные: файл-источник и название раздела
    (последний markdown-заголовок перед ним). Метаданные показываются
    пользователю как «источник ответа».
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    blocks: list[str] = []
    block_sections: list[str] = []
    current, current_section = "", ""
    for paragraph in paragraphs:
        first_line = paragraph.split("\n", 1)[0].strip()
        if first_line.startswith("#"):  # markdown-заголовок (или docx-заголовок из _load_docx)
            current_section = first_line.lstrip("# ").strip()
        if len(paragraph) > CHUNK_SIZE:
            if current:
                blocks.append(current)
                block_sections.append(current_section)
                current = ""
            for piece in _split_long_paragraph(paragraph):
                blocks.append(piece)
                block_sections.append(current_section)
            continue
        if current and len(current) + len(paragraph) + 1 > CHUNK_SIZE:
            blocks.append(current)
            block_sections.append(current_section)
            current = ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        blocks.append(current)
        block_sections.append(current_section)

    chunks: list[dict] = []
    prev_tail = ""
    for block, section in zip(blocks, block_sections):
        body = f"{prev_tail}\n{block}".strip() if prev_tail else block
        chunks.append({"text": body, "section": section or source})
        prev_tail = _tail(block, CHUNK_OVERLAP)
    return chunks


# --------------------------------------------------------------------------
# 3. Индексация в векторную БД
# --------------------------------------------------------------------------

def open_collection():
    """Открыть (или создать) коллекцию Chroma.

    anonymized_telemetry=False — не отправляем статистику наружу.
    cosine — мера близости векторов: чем меньше «расстояние», тем релевантнее кусок.
    """
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index() -> int:
    """Проиндексировать все документы из DOCS_DIR. Возвращает число кусков.

    Коллекция пересоздаётся с нуля: на демо-масштабе это проще и надёжнее,
    чем следить за изменениями отдельных файлов.
    """
    docs = load_documents()
    if not docs:
        print(f"В папке {DOCS_DIR} нет документов ({', '.join(sorted(SUPPORTED))}).")
        print("Положите файлы в папку и запустите индексацию снова.")
        return 0

    # Нарезка на куски
    chunks: list[dict] = []
    for doc in docs:
        for chunk in split_into_chunks(doc["text"], doc["source"]):
            chunks.append({"source": doc["source"], **chunk})
    print(f"Документов: {len(docs)} -> кусков после нарезки: {len(chunks)}")

    # Эмбеддинги — батчами, чтобы не слать сотни текстов одним запросом
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = [chunk["text"] for chunk in chunks[start:start + EMBED_BATCH]]
        vectors.extend(embed(batch))
        done = min(start + EMBED_BATCH, len(chunks))
        print(f"Эмбеддинги: {done}/{len(chunks)}")

    # Пересоздаём коллекцию и заливаем куски с метаданными
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # коллекции ещё нет — это нормально при первом запуске
    collection = open_collection()
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=vectors,
        metadatas=[
            {"source": chunk["source"], "section": chunk["section"]}
            for chunk in chunks
        ],
    )
    print(f"Готово: {collection.count()} кусков в базе «{COLLECTION_NAME}» ({DB_DIR})")
    return len(chunks)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # русский текст в консоли Windows
    build_index()
