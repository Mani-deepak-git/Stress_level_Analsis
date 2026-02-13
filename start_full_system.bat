@echo off
echo ========================================
echo Starting Complete Interview System
echo ========================================

REM Start AI Backend in new window
start "AI Backend" cmd /k "start_ai_backend.bat"

REM Wait 5 seconds for AI backend to initialize
timeout /t 5 /nobreak >nul

REM Start Node.js server in new window
start "Node Server" cmd /k "start_node_server.bat"

REM Wait 3 seconds for Node server to initialize
timeout /t 3 /nobreak >nul

REM Start React frontend in new window
start "React Frontend" cmd /k "cd frontend && npm start"

echo ========================================
echo All services starting...
echo AI Backend: http://localhost:8001
echo Node Server: http://localhost:3000
echo React Frontend: http://localhost:3001
echo ========================================
pause