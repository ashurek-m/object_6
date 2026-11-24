"""Веб-интерфейс RAG-ассистента (Streamlit).

Запуск:  streamlit run app.py   (или run_web.bat)
Откроется браузер на http://localhost:8501
"""

import sys

import streamlit as st

from config import LLM_BASE_URL, LLM_CHAT_MODEL, LLM_EMBED_MODEL, TOP_K
from ingest import build_index, open_collection
from query import answer_question

sys.stdout.reconfigure(encoding="utf-8")

st.set_page_config(page_title="RAG-ассистент", page_icon="🔍", layout="wide")

st.title("🔍 RAG-ассистент по производственной документации")
st.caption(
    "Отвечает на вопросы по регламентам и стандартам предприятия, указывая источник. "
    "Схема: эмбеддинги + векторный поиск (Chroma) + LLM. Локально через Ollama — документы не покидают компьютер."
)

# --- Боковая панель: подключение и обслуживание индекса ---
with st.sidebar:
    st.header("Настройки")
    top_k = st.slider("Сколько фрагментов подавать модели (top-K)", 1, 10, TOP_K)

    st.subheader("Подключение")
    st.text(f"API: {LLM_BASE_URL}")
    st.text(f"Модель ответов: {LLM_CHAT_MODEL}")
    st.text(f"Модель эмбеддингов: {LLM_EMBED_MODEL}")

    st.divider()
    if st.button("Переиндексировать документы"):
        with st.spinner("Индексирую документы..."):
            try:
                count = build_index()
            except RuntimeError as exc:
                st.error(f"Ошибка индексации: {exc}")
                count = None
        if count:
            st.success(f"Проиндексировано кусков: {count}")

# --- Проверка, что индекс вообще построен ---
try:
    indexed = open_collection().count()
except Exception:
    indexed = 0
if indexed == 0:
    st.info(
        "Индекс пуст. Положите документы (.md, .txt, .docx) в папку data/docs "
        "и нажмите «Переиндексировать документы» в боковой панели (или запустите python ingest.py)."
    )
    st.stop()

st.write(f"В базе проиндексировано фрагментов: **{indexed}**")

# --- Вопрос и ответ ---
question = st.text_input("Ваш вопрос", placeholder="Например: до какого времени принимают заявки на срочную отгрузку?")
ask = st.button("Спросить", type="primary")

if ask and question.strip():
    with st.spinner("Ищу релевантные фрагменты в документах..."):
        try:
            answer, chunks = answer_question(question.strip(), top_k)
        except RuntimeError as exc:
            st.error(f"Ошибка: {exc}")
            st.stop()

    st.subheader("Ответ")
    st.markdown(answer)

    st.subheader("Источники")
    for i, chunk in enumerate(chunks):
        similarity = 1 - chunk["distance"]
        with st.expander(f"[{i + 1}] {chunk['source']} — «{chunk['section']}» (близость: {similarity:.2f})"):
            st.text(chunk["text"])
elif ask:
    st.warning("Введите вопрос.")
