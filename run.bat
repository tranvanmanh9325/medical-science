@echo off
setlocal enabledelayedexpansion

title Apptronik Apollo 3D Biomechanics Simulation Lab
cd /d "%~dp0"

echo ==================================================================
echo  Apptronik Apollo Humanoid 3D Simulation ^& Telemetry Lab
echo ==================================================================

:: 1. Pre-launch cleanup: Don dep tat ca cac tien trinh python chay main.py cu de giai phong 100% GPU
echo [KHOI DONG] Dang quet va don dep tien trinh cu con sot lai...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process -Filter \"name = 'python.exe' or name = 'pythonw.exe'\" | " ^
  "Where-Object { $_.CommandLine -like '*main.py*' } | " ^
  "ForEach-Object { Write-Host '[DON DEP] Da dong tien trinh cu PID:' $_.ProcessId; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

:: 2. Khoi chay mo phong
echo [KHOI DONG] Dang khoi chay mo phong 3D MuJoCo...
python main.py
set EXIT_CODE=%ERRORLEVEL%

:: 3. Post-launch cleanup: Dam bao khong co luong chay ngam nao giu card GPU
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process -Filter \"name = 'python.exe' or name = 'pythonw.exe'\" | " ^
  "Where-Object { $_.CommandLine -like '*main.py*' } | " ^
  "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

if %EXIT_CODE% neq 0 (
    echo ==================================================================
    echo [CANH BAO] Chuong trinh ket thuc voi ma loi %EXIT_CODE%.
    echo ==================================================================
    pause
)
