# Chargement du fichier .env
$envPath = ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.+)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    Write-Host "Fichier .env non trouvé." -ForegroundColor Red
    exit 1
}

# Configuration automatique pour Windows
Write-Host "=== Configuration Docker pour Windows ==="
[System.Environment]::SetEnvironmentVariable("DOCKER_USER", "1000:1000", "Process")
[System.Environment]::SetEnvironmentVariable("PUID", "1000", "Process")
[System.Environment]::SetEnvironmentVariable("PGID", "1000", "Process")
Write-Host "Permissions Docker configurées pour Windows" -ForegroundColor Green


# Lancement de Docker Desktop
Write-Host "=== Lancement de Docker Desktop ==="
Start-Process "$env:DOCKER_DESKTOP_PATH"

# Attente du démarrage de Docker
Write-Host "=== Attente du démarrage de Docker... ==="
do {
    Start-Sleep -Seconds 3
    $dockerReady = & docker info 2>$null
} until ($dockerReady)
Write-Host "=== Docker est prêt ===" -ForegroundColor Green

# Activation de l'environnement Flask
Write-Host "=== Lancement de Flask ==="
Set-Location "$env:FLASK_DIR"

# Active le venv s'il existe
if (Test-Path "$env:VENV_PATH") {
    & $env:VENV_PATH\Scripts\Activate.ps1
    pip install -r requirements.txt
    python app.py
    # deactivate
} else {
    Write-Host "Environnement virtuel non trouvé." -ForegroundColor Yellow
    pip install -r requirements.txt
    python app.py
}