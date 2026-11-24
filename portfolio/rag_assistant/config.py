"""Конфигурация RAG-ассистента.

Все настройки читаются из файла .env (копия .env.example -> .env).
Благодаря этому один и тот же код работает:
  - локально через Ollama (по умолчанию) — данные не покидают компьютер;
  - с любым облачным OpenAI-совместимым API (OpenAI, OpenRouter и т.п.).
Достаточно поменять 2–3 строки в .env, код не трогаем.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта = папка, где лежит этот файл.
# Все относительные пути считаем от него, а не от текущей директории запуска.
ROOT = Path(__file__).resolve().parent

# .env лежит рядом с кодом; override=True — значения из файла важнее переменных окружения
load_dotenv(ROOT / ".env", override=True)


def _get(name: str, default: str) -> str:
    """Прочитать переменную из .env; пустое значение = использовать умолчание."""
    value = os.getenv(name, "").strip()
    return value if value else default


# --- Подключение к LLM (OpenAI-совместимый API) ---
LLM_BASE_URL = _get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _get("LLM_API_KEY", "ollama")  # для Ollama ключ не нужен, но поле должно быть непустым
LLM_CHAT_MODEL = _get("LLM_CHAT_MODEL", "qwen3:4b")   # модель для ответов
LLM_EMBED_MODEL = _get("LLM_EMBED_MODEL", "bge-m3")   # модель для эмбеддингов (хорошо работает с русским)

# --- Параметры RAG ---
CHUNK_SIZE = int(_get("CHUNK_SIZE", "900"))        # максимальный размер куска, символов
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))  # перекрытие между соседними кусками, символов
TOP_K = int(_get("TOP_K", "5"))                    # сколько кусков подавать модели на вопрос

# --- Пути ---
DOCS_DIR = ROOT / _get("DOCS_DIR", "data/docs")    # сюда кладут документы для индексации
DB_DIR = ROOT / _get("DB_DIR", "chroma_db")        # локальное векторное хранилище Chroma
COLLECTION_NAME = _get("COLLECTION_NAME", "rag_docs")
