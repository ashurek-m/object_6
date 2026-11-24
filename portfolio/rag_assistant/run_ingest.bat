@echo off
rem Индексация документов: data/docs -> чанки -> эмбеддинги -> chroma_db
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" ingest.py
pause
