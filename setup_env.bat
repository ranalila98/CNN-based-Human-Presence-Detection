@echo off
REM === Setup Conda Environment from environment.yml ===

echo Creating environment from environment.yml...
conda env create -f environment.yml

if errorlevel 1 (
    echo [!] Failed to create environment. Make sure environment.yml is valid.
    pause
    exit /b
)

echo Environment created successfully.

REM === Activate environment and launch a new shell ===
echo [*] Activating environment and opening new terminal...

call conda activate sdrcsi_env
cmd /k
