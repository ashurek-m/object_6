@echo off
rem Вопрос ассистенту из консоли: run_cli.bat "Как оформить отгрузку?"
rem Без аргументов открывается интерактивный режим (вопрос за вопросом).
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" query.py %*
