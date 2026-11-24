@echo off
rem Веб-интерфейс (Streamlit): http://localhost:8501
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\streamlit.exe" run app.py --server.address localhost --browser.gatherUsageStats false
