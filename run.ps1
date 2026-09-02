# Medical Robotics MuJoCo Simulation Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [1/2] Building Medical Robotics Docker Container..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

docker compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed. Please check Docker Desktop status." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [2/2] Launching 3D Simulation Window on Windows Desktop..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

docker compose run --rm robot-sim python main.py
