@echo off
call "%~dp0start.bat" --virtual-camera %*
exit /b %errorlevel%
