@echo off
cd /d "%~dp0"
echo ================================
echo  gemini-web2api starting...
echo  Base URL: http://localhost:8000/v1
echo  Band karne ke liye: Ctrl+C
echo ================================
python -m gemini_web2api --port 8000
pause
