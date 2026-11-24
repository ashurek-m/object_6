"""Мини-оценка качества ПОИСКА (retrieval evaluation).

Главный фильтр качества RAG — поиск: если нужный кусок не попал в топ-K,
ответ будет плохим при любой LLM. Поэтому сначала меряем именно поиск:
для каждого вопроса с известным ответом-«донором» проверяем, что
нужный документ попал в найденные.

После добавления своих документов дополните список CASES своими парами
«вопрос -> файл, где лежит ответ».

Запуск:  python eval.py
"""

import sys

from config import TOP_K
from query import retrieve

# Вопросы по демонстрационным документам из data/examples/.
# "expect" — часть имени файла, где лежит правильный ответ.
CASES = [
    {"question": "Что нужно, чтобы получить продукцию со склада?", "expect": "sto_otgruzka"},
    {"question": "До какого времени принимают заявки на срочную отгрузку?", "expect": "sto_otgruzka"},
    {"question": "Какие документы оформляются при отгрузке?", "expect": "sto_otgruzka"},
    {"question": "В какой срок покупатель может предъявить претензию по качеству?", "expect": "sto_otgruzka"},
    {"question": "До какого числа формируется месячный план производства?", "expect": "reglament_planirovaniya"},
    {"question": "Кто определяет приоритет заказов?", "expect": "reglament_planirovaniya"},
    {"question": "Что делать, если заказ сорвёт срок?", "expect": "reglament_planirovaniya"},
    {"question": "Какие позиции НЗП считаются зависшими?", "expect": "reglament_planirovaniya"},
    # «Негативный» вопрос: ответа нет ни в одном документе — проверяем честность поиска
    {"question": "Какой у компании ИНН и расчётный счёт?", "expect": ""},
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    hits = 0
    total = 0

    print(f"Оценка поиска: топ-{TOP_K} кусков на вопрос\n")
    for case in CASES:
        try:
            chunks = retrieve(case["question"], TOP_K)
        except RuntimeError as exc:
            print(f"Ошибка: {exc}")
            return

        found = [chunk["source"] for chunk in chunks]
        if case["expect"]:
            total += 1
            hit = any(case["expect"] in source for source in found)
            hits += hit
            mark = "OK " if hit else "ПРОМАХ"
            print(f"[{mark}] {case['question']}")
            print(f"       ожидали документ: {case['expect']}")
            print(f"       найдено: {', '.join(found)}")
        else:
            best = min(chunk["distance"] for chunk in chunks)
            print(f"[-- ] {case['question']} (ответа в корпусе нет)")
            print(f"       найдено: {', '.join(found)}")
            print(f"       лучшее расстояние: {best:.2f} — чем больше, тем меньше смысла в найденном")
        print()

    if total:
        percent = 100 * hits / total
        print(f"Итог: нужный документ найден в {hits} из {total} случаев ({percent:.0f}%)")


if __name__ == "__main__":
    main()
