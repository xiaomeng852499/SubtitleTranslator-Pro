@echo off
setlocal
cd /d "%~dp0"

if exist "launch_hidden.vbs" (
  wscript.exe "launch_hidden.vbs"
  exit /b 0
)

start "" /min pyw -3 "%~dp0jp_subtitle_translator.py" --gui
exit /b 0
