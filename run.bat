@echo off
title da Vinci Surgical Robot (dVRK PSM) Simulation
echo ==================================================================
echo  Launching da Vinci Surgical Robot 3D Simulation on Windows...
echo ==================================================================
python main.py
if errorlevel 1 (
    echo [INFO] Falling back to Docker runner...
    docker compose run --rm robot-sim python main.py
)
pause
