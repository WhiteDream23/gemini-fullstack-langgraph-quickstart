@echo off
title Qwen 智能研究助手
echo ========================================
echo       Qwen 智能研究助手启动脚本
echo ========================================
echo.

cd /d "%~dp0"

REM 检查虚拟环境
if exist "..\\.venv\\Scripts\\python.exe" (
    echo 使用虚拟环境...
    ..\.venv\Scripts\python.exe run_gradio.py
) else (
    echo 使用系统 Python...
    python run_gradio.py
)

echo.
echo 按任意键退出...
pause > nul
