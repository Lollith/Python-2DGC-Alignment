
import { displayMessage, viewLogs } from '../main.js';


export function initializeMonitoringTab() {
    const checkDockerBtn = document.getElementById('checkDockerBtn');
    const viewLogsBtn = document.getElementById('viewLogsBtn');
    const dockerStatusDiv = document.getElementById('dockerStatus');
    const refreshStatusBtn = document.getElementById('refreshStatusBtn');
    const restartDockerBtn = document.getElementById('restartDockerBtn');
    
    if (viewLogsBtn) {
        viewLogsBtn.addEventListener('click', viewLogs);
    }
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', refreshAllStatus);
    }

    if (restartDockerBtn) {
        restartDockerBtn.addEventListener('click', restartDocker);
    }

    async function checkDockerStatus(retries = 15, delayMs = 50000) {
        await new Promise(r => setTimeout(r, 500000));
        for (let i = 0; i < retries; i++) {
            try {
                const res = await fetch('/api/check_containers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                if (data.all_running) {
                    displayMessage('Docker est opérationnel', 'success');
                    return data; // Docker est up, on retourne les données
                }
            } catch (err) {
                console.log("Erreur fetch checkDockerStatus:", err);
            }
            await new Promise(r => setTimeout(r, delayMs));
        }
        return null;
    }

    // Vérifier l'état des conteneurs Docker
    checkDockerBtn.addEventListener('click', async function() {
        checkDockerBtn.disabled = true;
        checkDockerBtn.textContent = '🔄 Lancement Docker...';
        
        try {
            // 1 Vérifier si Docker est déjà lancé
            let response = await fetch('/api/check_containers', { method: 'POST' });
            let data = await response.json();

                if (!data.all_running) {
                // 2️.Lancer Docker seulement si nécessaire

                const response = await fetch('/api/start_containers', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                data = await checkDockerStatus();
            }
            
            if (data && data.all_running) {
                dockerStatusDiv.className = 'system-status docker-running';
                dockerStatusDiv.innerHTML = '🟢 Tous les conteneurs Docker sont en cours d\'exécution';
                displayMessage('Conteneurs Docker: Tous en cours d\'exécution');
            } else {
                dockerStatusDiv.className = 'system-status docker-stopped';
                dockerStatusDiv.innerHTML = '🔴 Certains conteneurs Docker ne sont pas en cours d\'exécution';
                displayMessage('Certains conteneurs Docker ne sont pas actifs', 'error');
            }
            if (data && data.status) {
                data.status.forEach(status => displayMessage(status, 'info'));
            }
            
        } catch (error) {
            displayMessage('Erreur lors de la vérification Docker: ' + error.message, 'error');
            dockerStatusDiv.className = 'system-status docker-stopped';
            dockerStatusDiv.innerHTML = '❌ Erreur lors de la vérification Docker';
        } finally {
            checkDockerBtn.disabled = false;
            checkDockerBtn.textContent = '🐳 Vérifier Docker';
        }
    });

    // Vérification initiale au chargement
    async function checkNistStatus() {
        try {
            const response = await fetch('/nist/health', { method: 'GET' });
            const data = await response.json();
            
            const nistStatusDiv = document.getElementById('nistStatus');
            
            if (data.nist_status === 'available') {
                nistStatusDiv.innerHTML = `
                    <div class="status-indicator status-running"></div>
                    🟢 Moteur NIST: Actif et prêt
                `;
                nistStatusDiv.className = 'system-status nist-running';
            } else {
                nistStatusDiv.innerHTML = `
                    <div class="status-indicator status-stopped"></div>
                    🔴 Moteur NIST: Indisponible
                `;
                nistStatusDiv.className = 'system-status nist-stopped';
            }
        } catch (error) {
            const nistStatusDiv = document.getElementById('nistStatus');
            nistStatusDiv.innerHTML = `
                <div class="status-indicator status-error"></div>
                ❌ Moteur NIST: Erreur de connexion
            `;
            nistStatusDiv.className = 'system-status nist-error';
        }
    }

    // actualiser tous les statuts
    async function refreshAllStatus() {
        const refreshBtn = document.getElementById('refreshStatusBtn');
        if (!refreshBtn) return;
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '🔄 <span>Actualisation...</span>';
        
        try {
            // Vérifier Docker
            const dockerResponse = await fetch('/api/check_containers', { method: 'POST' });
            const dockerData = await dockerResponse.json();
            
            const dockerStatusDiv = document.getElementById('dockerStatus');
            if (dockerData.all_running) {
                dockerStatusDiv.innerHTML = `
                    <div class="status-indicator status-running"></div>
                    🟢 Tous les conteneurs Docker sont en cours d'exécution
                `;
                dockerStatusDiv.className = 'system-status docker-running';
            } else {
                dockerStatusDiv.innerHTML = `
                    <div class="status-indicator status-stopped"></div>
                    🔴 Certains conteneurs Docker sont arrêtés
                `;
                dockerStatusDiv.className = 'system-status docker-stopped';
            }
            await checkNistStatus();
            } catch (error) {
                console.error('Erreur lors de l\'actualisation:', error);
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '🔄 <span>Actualiser l\'état</span>';
            }
        }

// Event listener pour le bouton actualiser
document.getElementById('refreshStatusBtn')?.addEventListener('click', refreshAllStatus);

    // redémarrer Docker
    async function restartDocker() {
        const restartDockerBtn = document.getElementById('restartDockerBtn');
        if (!restartDockerBtn) return;
        
        restartDockerBtn.disabled = true;
        restartDockerBtn.innerHTML = '🔄 <span>Redémarrage...</span>';
        
        try {
            const response = await fetch('/api/restart_containers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            
            if (data.success) {
                displayMessage('🔄 Redémarrage des conteneurs Docker...', 'info');
                
                // Afficher TOUS les messages de restart
                data.status?.forEach(msg => {
                    const isError = msg.includes('❌');
                    const isSuccess = msg.includes('✨');
                    const type = isError ? 'error' : (isSuccess ? 'success' : 'info');
                    displayMessage(msg, type);
                });
                
            } else {
                displayMessage('❌ Erreur lors du redémarrage de Docker', 'error');
                data.status?.forEach(msg => displayMessage(msg, 'error'));
            }
            
        } catch (error) {
            displayMessage('❌ Erreur de connexion: ' + error.message, 'error');
        } finally {
            restartDockerBtn.disabled = false;
            restartDockerBtn.innerHTML = '🔄 <span>Redémarrer Docker</span>';
        }
    }
}