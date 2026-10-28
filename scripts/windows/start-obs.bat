@echo off
call "%~dp0start.bat" --no-virtual-camera %*
exit /b %errorlevel%
