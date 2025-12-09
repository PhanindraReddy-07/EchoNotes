# EchoNotes Windows Setup Script
# Run: .\setup.ps1

Write-Host "EchoNotes Setup for Windows" -ForegroundColor Cyan
Write-Host "=" * 40

# Check Python
Write-Host "`nChecking Python..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found. Install from python.org" -ForegroundColor Red
    exit 1
}
python --version

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path .venv) {
    Write-Host "Virtual environment already exists"
} else {
    python -m venv .venv
}

# Activate
Write-Host "`nActivating environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path output | Out-Null
New-Item -ItemType Directory -Force -Path samples | Out-Null

# Download models
Write-Host "`nDownloading models (this may take a few minutes)..." -ForegroundColor Yellow
python setup_models.py --small

Write-Host "`n" + "=" * 40 -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Open folder in VS Code: code ."
Write-Host "  2. Press F5 to run with debugging"
Write-Host "  3. Or run: python main.py --record --duration 30"
