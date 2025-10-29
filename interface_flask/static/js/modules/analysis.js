import { displayMessage, loadingDiv } from '../main.js';


export function initializeAnalysisTab() {
    const listH5Btn = document.getElementById('listH5Btn');
    // const checkDockerBtn = document.getElementById('checkDockerBtn');
    const analysisForm = document.getElementById('analysisForm');
    const h5FilesSelect = document.getElementById('h5Files');
    const dockerStatusDiv = document.getElementById('dockerStatus');
    // const viewLogsBtn = document.getElementById('viewLogsBtn');
    // if (viewLogsBtn) {
    //     viewLogsBtn.addEventListener('click', viewLogs);
    // }
    // const refreshStatusBtn = document.getElementById('refreshStatusBtn');
    // if (refreshStatusBtn) {
    //     refreshStatusBtn.addEventListener('click', refreshAllStatus);
    // }

    // const restartDockerBtn = document.getElementById('restartDockerBtn');
    // if (restartDockerBtn) {
    //     restartDockerBtn.addEventListener('click', restartDocker);
    // }

    listH5Btn.addEventListener('click', async function() {
            const analysisPath = document.getElementById('analysisPath').value;
            
            if (!analysisPath.trim()) {
                displayMessage('Veuillez spécifier un chemin pour les fichiers .h5', 'error');
                return;
            }
            
            listH5Btn.disabled = true;
            listH5Btn.textContent = '🔄 Chargement...';
            
            try {
                const response = await fetch('/api/list_files', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ path: analysisPath, extension: '.h5' })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    h5FilesSelect.innerHTML = '';
                    if (data.files.length > 0) {
                        data.files.forEach(file => {
                            const option = document.createElement('option');
                            option.value = file;
                            option.textContent = file;
                            h5FilesSelect.appendChild(option);
                        });
                        displayMessage(`${data.files.length} fichier(s) .h5 trouvé(s)`);
                    } else {
                        const option = document.createElement('option');
                        option.textContent = 'Aucun fichier .h5 trouvé';
                        option.disabled = true;
                        h5FilesSelect.appendChild(option);
                        displayMessage('Aucun fichier .h5 trouvé', 'error');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                }
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listH5Btn.disabled = false;
                listH5Btn.textContent = '📋 Lister fichiers HDF5';
            }
        });
        
//         async function checkDockerStatus(retries = 15, delayMs = 50000) {
//             await new Promise(r => setTimeout(r, 500000));
//             for (let i = 0; i < retries; i++) {
//                 try {
//                     const res = await fetch('/api/check_containers', {
//                         method: 'POST',
//                         headers: { 'Content-Type': 'application/json' }
//                     });
//                     const data = await res.json();
//                     if (data.all_running) {
//                         displayMessage('Docker est opérationnel', 'success');
//                         return data; // Docker est up, on retourne les données
//                     }
//                 } catch (err) {
//                     console.log("Erreur fetch checkDockerStatus:", err);
//                 }
//                 await new Promise(r => setTimeout(r, delayMs));
//             }
//             return null;
//         }

//         // Vérifier l'état des conteneurs Docker
//         checkDockerBtn.addEventListener('click', async function() {
//             checkDockerBtn.disabled = true;
//             checkDockerBtn.textContent = '🔄 Lancement Docker...';
            
//             try {
//                 // 1 Vérifier si Docker est déjà lancé
//                 let response = await fetch('/api/check_containers', { method: 'POST' });
//                 let data = await response.json();

//                  if (!data.all_running) {
//                    // 2️.Lancer Docker seulement si nécessaire

//                     const response = await fetch('/api/start_containers', {
//                         method: 'POST',
//                         headers: {
//                             'Content-Type': 'application/json',
//                         }
//                     });
//                     data = await checkDockerStatus();
//                 }
                
//                 if (data && data.all_running) {
//                     dockerStatusDiv.className = 'system-status docker-running';
//                     dockerStatusDiv.innerHTML = '🟢 Tous les conteneurs Docker sont en cours d\'exécution';
//                     displayMessage('Conteneurs Docker: Tous en cours d\'exécution');
//                 } else {
//                     dockerStatusDiv.className = 'system-status docker-stopped';
//                     dockerStatusDiv.innerHTML = '🔴 Certains conteneurs Docker ne sont pas en cours d\'exécution';
//                     displayMessage('Certains conteneurs Docker ne sont pas actifs', 'error');
//                 }
//                 if (data && data.status) {
//                     data.status.forEach(status => displayMessage(status, 'info'));
//                 }
                
//             } catch (error) {
//                 displayMessage('Erreur lors de la vérification Docker: ' + error.message, 'error');
//                 dockerStatusDiv.className = 'system-status docker-stopped';
//                 dockerStatusDiv.innerHTML = '❌ Erreur lors de la vérification Docker';
//             } finally {
//                 checkDockerBtn.disabled = false;
//                 checkDockerBtn.textContent = '🐳 Vérifier Docker';
//             }
//         });

//         // checkNistStatus(); // Vérification initiale au chargement
//         async function checkNistStatus() {
//             try {
//                 const response = await fetch('/nist/health', { method: 'GET' });
//                 const data = await response.json();
                
//                 const nistStatusDiv = document.getElementById('nistStatus');
                
//                 if (data.nist_status === 'available') {
//                     nistStatusDiv.innerHTML = `
//                         <div class="status-indicator status-running"></div>
//                         🟢 Moteur NIST: Actif et prêt
//                     `;
//                     nistStatusDiv.className = 'system-status nist-running';
//                 } else {
//                     nistStatusDiv.innerHTML = `
//                         <div class="status-indicator status-stopped"></div>
//                         🔴 Moteur NIST: Indisponible
//                     `;
//                     nistStatusDiv.className = 'system-status nist-stopped';
//                 }
//             } catch (error) {
//                 const nistStatusDiv = document.getElementById('nistStatus');
//                 nistStatusDiv.innerHTML = `
//                     <div class="status-indicator status-error"></div>
//                     ❌ Moteur NIST: Erreur de connexion
//                 `;
//                 nistStatusDiv.className = 'system-status nist-error';
//             }
//         }

// // actualiser tous les statuts
//         async function refreshAllStatus() {
//             const refreshBtn = document.getElementById('refreshStatusBtn');
//             if (!refreshBtn) return;
//             refreshBtn.disabled = true;
//             refreshBtn.innerHTML = '🔄 <span>Actualisation...</span>';
            
//             try {
//                 // Vérifier Docker
//                 const dockerResponse = await fetch('/api/check_containers', { method: 'POST' });
//                 const dockerData = await dockerResponse.json();
                
//                 const dockerStatusDiv = document.getElementById('dockerStatus');
//                 if (dockerData.all_running) {
//                     dockerStatusDiv.innerHTML = `
//                         <div class="status-indicator status-running"></div>
//                         🟢 Tous les conteneurs Docker sont en cours d'exécution
//                     `;
//                     dockerStatusDiv.className = 'system-status docker-running';
//                 } else {
//                     dockerStatusDiv.innerHTML = `
//                         <div class="status-indicator status-stopped"></div>
//                         🔴 Certains conteneurs Docker sont arrêtés
//                     `;
//                     dockerStatusDiv.className = 'system-status docker-stopped';
//                 }
                
//                 // Vérifier NIST
//                 await checkNistStatus();
        
//                 } catch (error) {
//                     console.error('Erreur lors de l\'actualisation:', error);
//                 } finally {
//                     refreshBtn.disabled = false;
//                     refreshBtn.innerHTML = '🔄 <span>Actualiser l\'état</span>';
//                 }
//             }

// // Event listener pour le bouton actualiser
// document.getElementById('refreshStatusBtn')?.addEventListener('click', refreshAllStatus);


// Lancer l'analyse
        analysisForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const analysisPath = document.getElementById('analysisPath').value;
            const selectedFiles = Array.from(h5FilesSelect.selectedOptions).map(option => option.value);
            loadingDiv.style.display = 'block';
            // outputDiv.innerHTML = '';
            
            const data = {
                analysis_path: analysisPath,
                selected_files: selectedFiles
            };
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Afficher tous les messages
                result.messages.forEach(msg => {
                    const isError = msg.toLowerCase().includes('erreur');
                    displayMessage(msg, isError ? 'error' : 'success');
                });
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                loadingDiv.style.display = 'none';
            }
        });


        // Vérifier l'état Docker au chargement de la page
        window.addEventListener('load', function() {
            // Vérification automatique discrète
            fetch('/api/check_containers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            }).then(response => response.json())
            .then(data => {
                if (data.all_running) {
                    dockerStatusDiv.className = 'system-status docker-running';
                    dockerStatusDiv.innerHTML = '🟢 Tous les conteneurs Docker sont en cours d\'exécution';
                } else {
                    dockerStatusDiv.className = 'system-status docker-stopped';
                    dockerStatusDiv.innerHTML = '🔴 Certains conteneurs Docker ne sont pas en cours d\'exécution';
                }
            }).catch(() => {
                dockerStatusDiv.className = 'system-status docker-stopped';
                dockerStatusDiv.innerHTML = '❌ Impossible de vérifier l\'état Docker';
            });
        });
    }
