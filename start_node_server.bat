@echo off
echo ========================================
echo Starting Node.js WebRTC Server
echo ========================================

cd backend\node_server

REM Install dependencies if needed
if not exist node_modules (
    echo Installing Node.js dependencies...
    npm install
)

REM Start Node.js server
node server.js

pause