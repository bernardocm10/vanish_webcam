@echo off
call "%~dp0start.bat" --virtual-camera --call-audio %*
exit /b %errorlevel%
