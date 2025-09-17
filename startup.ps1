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

# verifier que wsl est accessible
wsl --status > $null 2>&1
if ($LASTEXITCODE -ne 0){
    Write-Host "❌WSL n'est pas disponible. Docker Desktop 
    ne pourra pas demarrer." -ForegroundColor Red
    exit 1
}

# Lancement de Docker Desktop / verifie que docker peut utiliser WSL
Write-Host "=== Lancement de Docker Desktop ==="
Start-Process "$env:DOCKER_DESKTOP_PATH"
$dockerReady = $false
do {
    Start-Sleep -Seconds 3
    docker info > $null 2>&1
    if ($LASTEXITCODE -eq 0){$dockerReady =$true}
    else {Write-Host "Docker Desktop n'est pas encore pret ou WSL
    n'est pas compatible..."}
} until ($dockerReady)

Write-Host "=== Docker est pret ==="

# Nettoyage des serveurs Flask existants (port 8080)
Write-Host "=== Verification des serveurs Flask sur le port 8080 ==="
$port = 8080
$flaskPids = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique

if ($flaskPids) {
    foreach ($procId in $flaskPids) {
        try {
            Write-Host "Arrêt du serveur Flask sur le port $port (PID=$procId)"
            Stop-Process -Id $procId -Force
        } catch {
            Write-Host "Impossible de stopper PID=$procId" -ForegroundColor Red
        }
    }
} else {
    Write-Host "Aucun Flask actif sur le port $port."
}

# Nettoyage des serveurs Flask existants (port 8080)
Write-Host "=== Verification des serveurs Flask sur le port 8080 ==="
$port = 8080
$flaskPids = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique

if ($flaskPids) {
    foreach ($procId in $flaskPids) {
        try {
            Write-Host "Arrêt du serveur Flask sur le port $port (PID=$procId)"
            Stop-Process -Id $procId -Force
        } catch {
            Write-Host "Impossible de stopper PID=$procId" -ForegroundColor Red
        }
    }
} else {
    Write-Host "Aucun Flask actif sur le port $port."
}

# Activation de l'environnement Flask
Write-Host "=== Lancement de Flask ==="
Set-Location "$env:PROJECT_PATH\interface_flask"

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