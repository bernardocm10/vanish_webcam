@echo off
setlocal
cd /d "%~dp0..\.."

rem Messages follow the Windows display language, like the app; VANISHCAM_LANG=pt or en overrides it.
if not defined VANISHCAM_LANG (
  for /f "delims=" %%L in ('powershell -NoProfile -Command "(Get-UICulture).TwoLetterISOLanguageName" 2^>nul') do set "VANISHCAM_LANG=%%L"
)
if /i "%VANISHCAM_LANG%"=="pt" (
  set "MSG_VENV=Criando o ambiente Python local..."
  set "MSG_INSTALL=Instalando os componentes necessarios. A internet so e necessaria desta vez..."
  set "MSG_START=Abrindo a webcam..."
  set "MSG_ERROR=Nao foi possivel abrir o VanishCam. Veja docs\configuration.md para as solucoes mais comuns."
) else (
  set "MSG_VENV=Creating the local Python environment..."
  set "MSG_INSTALL=Installing the required components. Internet is needed only this time..."
  set "MSG_START=Starting the webcam..."
  set "MSG_ERROR=Could not start VanishCam. See docs\configuration.md for the most common fixes."
)

set "VENV_DIR=.venv-local"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] %MSG_VENV%
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv "%VENV_DIR%"
  ) else (
    python -m venv "%VENV_DIR%"
  )
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam, pygame, PIL" >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] %MSG_INSTALL%
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

echo [VanishCam] %MSG_START%
"%VENV_PY%" -m vanishcam %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo %MSG_ERROR%
pause
exit /b 1
