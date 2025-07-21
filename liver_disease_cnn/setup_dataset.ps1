# 🏥 Liver Disease Dataset Setup PowerShell Script
Write-Host "🏥 Liver Disease Dataset Setup" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green

# Check for Python
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
$pythonCmd = $null

try {
    python --version | Out-Null
    $pythonCmd = "python"
    Write-Host "✅ Found: python" -ForegroundColor Green
} catch {
    try {
        py --version | Out-Null
        $pythonCmd = "py"
        Write-Host "✅ Found: py launcher" -ForegroundColor Green
    } catch {
        Write-Host "❌ Python not found!" -ForegroundColor Red
        Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Create directory structure
Write-Host "`n📁 Creating directory structure..." -ForegroundColor Yellow
$splits = @("train", "val", "test")
$classes = @("normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis")

foreach ($split in $splits) {
    foreach ($class in $classes) {
        $dir = "data\$split\$class"
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

# Check dataset status
Write-Host "`n📊 Dataset Status:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan

if (Test-Path "downloads\indian_liver_patient.csv") {
    Write-Host "✅ Found: indian_liver_patient.csv" -ForegroundColor Green
} else {
    Write-Host "❌ Missing: indian_liver_patient.csv" -ForegroundColor Red
}

if (Test-Path "downloads\medical_images.zip") {
    Write-Host "✅ Found: medical_images.zip" -ForegroundColor Green
} else {
    Write-Host "❌ Missing: medical_images.zip" -ForegroundColor Red
}

# Check current data
Write-Host "`n📁 Current Data Directory:" -ForegroundColor Yellow
$totalImages = 0
foreach ($split in $splits) {
    Write-Host "`n$($split.ToUpper()):" -ForegroundColor White
    foreach ($class in $classes) {
        $dir = "data\$split\$class"
        if (Test-Path $dir) {
            $count = (Get-ChildItem -Path $dir -Filter "*.jpg" | Measure-Object).Count
            Write-Host "  $($class.PadRight(15)): $count images" -ForegroundColor Gray
            $totalImages += $count
        }
    }
}

Write-Host "`nTotal current images: $totalImages" -ForegroundColor Cyan

# Run Python script to create enhanced dataset
Write-Host "`n🎨 Creating enhanced medical dataset..." -ForegroundColor Yellow
try {
    & $pythonCmd complete_dataset_setup.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Dataset setup complete!" -ForegroundColor Green
        Write-Host "`n🚀 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Open Jupyter Notebook: jupyter lab" -ForegroundColor White
        Write-Host "2. Navigate to: notebooks/liver_disease_classification.ipynb" -ForegroundColor White
        Write-Host "3. Run the cells to train your model!" -ForegroundColor White
        Write-Host "`n📚 Documentation:" -ForegroundColor Cyan
        Write-Host "- README.md: Project overview" -ForegroundColor White
        Write-Host "- GETTING_STARTED.md: Step-by-step guide" -ForegroundColor White
        Write-Host "- KAGGLE_SETUP.md: Kaggle integration" -ForegroundColor White
    } else {
        Write-Host "`n❌ Dataset setup failed!" -ForegroundColor Red
        Write-Host "Check error messages above." -ForegroundColor Yellow
    }
} catch {
    Write-Host "`n❌ Error running Python script:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host "`nPress Enter to continue..." -ForegroundColor Yellow
Read-Host
