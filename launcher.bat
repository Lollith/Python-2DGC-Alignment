@echo off


echo === Lancement de Docker Desktop ===
start "" "C:\ProgramData\Microsoft\Windows\Start Menu\Docker Desktop"
REM Attendre que Docker soit prêt (optionnel mais recommandé)
echo === Attente du démarrage de Docker... ===
:waitloop
docker info >nul 2>&1
if errorlevel 1 (
    timeout /t 3 >nul
    goto waitloop
)

echo === Docker est prêt ===
echo === Lancement de Flask ===
cd "D:\Dossiers Persos\Adeline\Python-2DGC-Alignment\interface_flask"
pip install -r requirements.txt
python app.py
pause