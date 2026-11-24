"""ЭТАП 2 RAG — ответ на вопрос (запускается каждый раз, когда задают вопрос).

Пайплайн запроса:
    вопрос -> эмбеддинг -> векторный поиск топ-K кусков -> промпт с фрагментами
           -> LLM отвечает со ссылками на источники [1], [2]...

Запуск:
    python query.py "Как оформить отгрузку?"   # один вопрос
    python query.py                            # интерактивный режим
"""

import argparse
import sys

from config import TOP_K
from ingest import open_collection
from llm_client import embed

# Системный промпт — главный «предохранитель» от галлюцинаций:
# модели прямо запрещено отвечать вне фрагментов.
SYSTEM_PROMPT = """Ты — ассистент по производственной документации предприятия.
Отвечай на вопрос ТОЛЬКО по приведённым фрагментам документов.
Правила:
1. Каждый используемый факт помечай номером фрагмента в квадратных скобках: [1], [2].
2. Если ответа во фрагментах нет — прямо скажи: «В предоставленных документах ответа нет». Ничего не выдумывай.
3. Отвечай кратко и по делу, на русском языке. Точные значения (сроки, номера, лимиты) бери дословно из фрагментов.
"""


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Найти top_k кусков, ближайших к вопросу по смыслу.

    Вопрос и куски сравниваются как векторы (эмбеддинги):
    косинусное расстояние мало -> смысл близок -> кусок релевантен.
    """
    collection = open_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "Индекс пуст — сначала запустите индексацию: python ingest.py"
        )

    result = collection.query(
        query_embeddings=embed([question]),
        n_results=top_k,
    )
    chunks: list[dict] = []
    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "section": meta.get("section", ""),
            "distance": distance,  # косинусное расстояние: 0 = идентичны, 2 = противоположны
        })
    return chunks


def answer_question(question: str, top_k: int = TOP_K) -> tuple[str, list[dict]]:
    """Полный цикл RAG: поиск -> промпт -> ответ LLM. Возвращает ответ и куски."""
    chunks = retrieve(question, top_k)
    fragments = "\n\n".join(
        f"[{i + 1}] Источник: {chunk['source']}, раздел «{chunk['section']}»\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )
    user_prompt = f"Фрагменты документов:\n\n{fragments}\n\nВопрос: {question}"
    answer = chat_with_sources(user_prompt)
    return answer, chunks


def chat_with_sources(user_prompt: str) -> str:
    """Обёртка на случай, когда из app.py импортируют чистый вызов LLM."""
    from llm_client import chat  # локальный импорт — чтобы модуль грузился быстрее
    return chat(SYSTEM_PROMPT, user_prompt)


def print_answer(answer: str, chunks: list[dict]) -> None:
    """Красиво вывести ответ и список источников в консоль."""
    print("\n" + "=" * 70)
    print("ОТВЕТ")
    print("=" * 70)
    print(answer)

    print("\n" + "-" * 70)
    print("ИСТОЧНИКИ")
    print("-" * 70)
    for i, chunk in enumerate(chunks):
        similarity = 1 - chunk["distance"]  # переводим расстояние в «близость» 0..1
        print(f"[{i + 1}] {chunk['source']} — «{chunk['section']}» (близость: {similarity:.2f})")
        if SHOW_FULL_CHUNKS:
            print("    " + chunk["text"][:600].replace("\n", "\n    "))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="RAG-ассистент: вопрос -> ответ с источниками")
    parser.add_argument("question", nargs="*", help="вопрос (если не задан — интерактивный режим)")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="сколько кусков искать")
    parser.add_argument("--full", action="store_true", help="показывать текст найденных кусков")
    args = parser.parse_args()

    global SHOW_FULL_CHUNKS
    SHOW_FULL_CHUNKS = args.full

    if args.question:
        try:
            answer, chunks = answer_question(" ".join(args.question), args.top_k)
            print_answer(answer, chunks)
        except RuntimeError as exc:
            print(f"Ошибка: {exc}")
        return

    # Интерактивный режим: задавайте вопросы один за другим
    print("RAG-ассистент по производственной документации. Пустая строка — выход.")
    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            answer, chunks = answer_question(question, args.top_k)
            print_answer(answer, chunks)
        except RuntimeError as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
