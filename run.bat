@echo off
cd /d "%~dp0"
echo Starting FastAPI backend...
start "CropPrice API" cmd /k "python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo Starting Streamlit dashboard...
start "CropPrice Dashboard" cmd /k "python -m streamlit run dashboard\app.py --server.port 8501"
echo.
echo Dashboard: http://localhost:8501
echo API docs:  http://localhost:8000/docs
