"""HTTP-клиент к OpenAI-совместимому API.

Написан на «голом» requests, а не на SDK, чтобы было видно, какие
эндпоинты вызываются и что приходит в ответе: удобно и для отладки,
и для объяснения на собеседовании. Один и тот же код работает с Ollama
(localhost), OpenAI, OpenRouter и любым другим OpenAI-совместимым API —
адрес и ключ берутся из .env (см. config.py).
"""

import re

import requests

from config import LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_MODEL, LLM_EMBED_MODEL

TIMEOUT = 300  # секунд; локальная модель на слабом железе может отвечать медленно


def _post(path: str, payload: dict) -> dict:
    """POST-запрос к API с понятной диагностикой ошибок."""
    try:
        response = requests.post(
            f"{LLM_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к {LLM_BASE_URL}. "
            "Если используете Ollama — запустите его и проверьте LLM_BASE_URL в .env"
        ) from exc

    if response.status_code >= 400:
        hint = {
            401: "проверьте LLM_API_KEY в .env",
            403: "проверьте LLM_API_KEY в .env",
            404: "проверьте LLM_BASE_URL (должен заканчиваться на /v1) "
                 "и имена моделей в .env — возможно, модель не установлена",
        }.get(response.status_code, "")
        raise RuntimeError(
            f"API вернул ошибку {response.status_code} ({path}). {hint}. "
            f"Ответ сервера: {response.text[:300]}"
        )
    return response.json()


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """Задать вопрос LLM. Возвращает текст ответа.

    temperature 0.2 — почти детерминированный режим: для ответов
    по документам «фантазия» не нужна.
    """
    data = _post(
        "/chat/completions",
        {
            "model": LLM_CHAT_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    content = data["choices"][0]["message"]["content"] or ""

    # Некоторые модели (например, qwen3 в Ollama) по умолчанию пишут
    # «размышления» в тегах <think>...</think> прямо в ответе.
    # Для демо нужен чистый текст — вырезаем.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    return content.strip()


def embed(texts: list[str]) -> list[list[float]]:
    """Превратить список текстов в эмбеддинги (векторы смысла).

    Близкие по смыслу тексты -> близкие векторы: на этом построен поиск.
    """
    data = _post("/embeddings", {"model": LLM_EMBED_MODEL, "input": texts})
    items = sorted(data["data"], key=lambda item: item["index"])  # порядок как на входе
    return [item["embedding"] for item in items]
