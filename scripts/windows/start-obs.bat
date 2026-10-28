@echo off
call "%~dp0start.bat" --no-virtual-camera --video-window %*
exit /b %errorlevel%
