@echo off
echo ========================================
echo Starting AI Backend Server
echo ========================================

REM Activate Python environment
call ai_env\Scripts\activate.bat

REM Start FastAPI server
cd backend\ai_server
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

pause