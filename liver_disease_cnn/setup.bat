@echo off
REM Quick Setup Script for Liver Disease Classification System (Windows)
REM Run this script to quickly set up your environment

echo 🚀 Liver Disease Classification System - Windows Setup
echo ================================================================

echo.
echo 📋 Step 1: Installing Python Dependencies...
echo ================================================================
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ❌ Failed to upgrade pip. Please check your Python installation.
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install packages. Check requirements.txt and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!

echo.
echo 📁 Step 2: Setting up directory structure...
echo ================================================================
python setup.py
if %errorlevel% neq 0 (
    echo ❌ Setup script failed. Please check the error above.
    pause
    exit /b 1
)

echo.
echo 🎉 Setup Complete!
echo ================================================================
echo.
echo 📋 What's Next:
echo    1. Add your liver images to data/ folders
echo    2. Open Jupyter: jupyter lab notebooks/liver_disease_classification.ipynb
echo    3. Run all cells to train your model
echo    4. Launch web interface: python app/gradio_app.py
echo.
echo ⚠️  IMPORTANT: This is for research/educational use only!
echo    Always consult medical professionals for actual diagnosis.
echo.
echo 📚 See GETTING_STARTED.md for detailed instructions
echo.

pause
