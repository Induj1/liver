@echo off
echo 🏥 Liver Disease Dataset Setup
echo ===============================

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH. Trying py launcher...
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python not installed or not in PATH
        echo Please install Python from https://python.org
        pause
        exit /b 1
    )
    echo Using py launcher
    set PYTHON_CMD=py
) else (
    echo Using python command
    set PYTHON_CMD=python
)

echo.
echo Checking required packages...
%PYTHON_CMD% -c "import PIL, numpy, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    %PYTHON_CMD% -m pip install pillow numpy pandas
)

REM Create directory structure
mkdir data\train\normal 2>nul
mkdir data\train\cirrhosis 2>nul
mkdir data\train\liver_cancer 2>nul
mkdir data\train\fatty_liver 2>nul
mkdir data\train\hepatitis 2>nul

mkdir data\val\normal 2>nul
mkdir data\val\cirrhosis 2>nul
mkdir data\val\liver_cancer 2>nul
mkdir data\val\fatty_liver 2>nul
mkdir data\val\hepatitis 2>nul

mkdir data\test\normal 2>nul
mkdir data\test\cirrhosis 2>nul
mkdir data\test\liver_cancer 2>nul
mkdir data\test\fatty_liver 2>nul
mkdir data\test\hepatitis 2>nul

echo.
echo 📊 Dataset Status:
echo ==================

echo Checking downloads...
if exist "downloads\indian_liver_patient.csv" (
    echo ✅ Found: indian_liver_patient.csv
) else (
    echo ❌ Missing: indian_liver_patient.csv
)

if exist "downloads\medical_images.zip" (
    echo ✅ Found: medical_images.zip
) else (
    echo ❌ Missing: medical_images.zip
)

echo.
echo 🎨 Creating enhanced medical dataset...
%PYTHON_CMD% complete_dataset_setup.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Dataset setup complete!
    echo.
    echo 🚀 Next steps:
    echo 1. Open Jupyter Notebook: jupyter lab
    echo 2. Navigate to: notebooks/liver_disease_classification.ipynb
    echo 3. Run the cells to train your model!
    echo.
    echo 📚 Documentation:
    echo - README.md: Project overview
    echo - GETTING_STARTED.md: Step-by-step guide
    echo - KAGGLE_SETUP.md: Kaggle integration
) else (
    echo.
    echo ❌ Dataset setup failed!
    echo Check error messages above.
)

echo.
pause
                x = random.randint(20, 200)
                y = random.randint(20, 200)
                r = random.randint(5, 15)
                shade = tuple(max(0, c - random.randint(0, 30)) for c in colors[class_name])
                draw.ellipse([x-r, y-r, x+r, y+r], fill=shade)
            
            # Save image
            img_path = class_dir / f'demo_{class_name}_{i:03d}.jpg'
            img.save(img_path, 'JPEG')
            total += 1

print(f'Created {total} demo images successfully!')
"

echo.
echo Demo dataset created with %total% images!
echo.
echo Directory structure:
echo data/train/ - Training images (15 per class)
echo data/val/   - Validation images (5 per class)  
echo data/test/  - Test images (5 per class)
echo.
echo Classes: normal, cirrhosis, liver_cancer, fatty_liver, hepatitis
echo.
echo Next step: Run 'jupyter lab notebooks/liver_disease_classification.ipynb'
pause
