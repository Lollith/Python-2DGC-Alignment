import { displayMessage, loadingDiv } from '../main.js';


export function initializeAnalysisTab() {
    const listH5Btn = document.getElementById('listH5Btn');
    const analysisForm = document.getElementById('analysisForm');
    const h5FilesSelect = document.getElementById('h5Files');
    const dockerStatusDiv = document.getElementById('dockerStatus');
    const availableFilesDiv = document.getElementById('availableFilesH5');
    

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
                
        //         if (data.success) {
        //             h5FilesSelect.innerHTML = '';
        //             if (data.files.length > 0) {
        //                 data.files.forEach(file => {
        //                     const option = document.createElement('option');
        //                     option.value = file;
        //                     option.textContent = file;
        //                     h5FilesSelect.appendChild(option);
        //                 });
        //                 displayMessage(`${data.files.length} fichier(s) .h5 trouvé(s)`);
        //             } else {
        //                 const option = document.createElement('option');
        //                 option.textContent = 'Aucun fichier .h5 trouvé';
        //                 option.disabled = true;
        //                 h5FilesSelect.appendChild(option);
        //                 displayMessage('Aucun fichier .h5 trouvé', 'error');
        //             }
        //         } else {
        //             displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
        //         }
        //     } catch (error) {
        //         displayMessage('Erreur de connexion: ' + error.message, 'error');
        //     } finally {
        //         listH5Btn.disabled = false;
        //         listH5Btn.textContent = '📋 Lister fichiers HDF5';
        //     }
        // });
            
                if (data.success) {
                    if (data.messages && data.messages.length > 0) {
                        data.messages.forEach(msg => {
                        const isWarning = msg.includes('⚠️') || msg.includes('ATTENTION');
                        displayMessage(msg, isWarning ? 'warning' : 'info');
                        });
                    }
                    if (data.files.length > 0) {
                        // ✅ Affichage groupé par dossier
                        const filesByFolder = {};
                        data.files.forEach(file => {
                            if (file.includes('/')) {
                                const parts = file.split('/');
                                const folder = parts.slice(0, -1).join('/');
                                const filename = parts[parts.length - 1];
                                if (!filesByFolder[folder]) {
                                    filesByFolder[folder] = [];
                                }
                                filesByFolder[folder].push(filename);
                            } else {
                                if (!filesByFolder['root']) {
                                    filesByFolder['root'] = [];
                                }
                                filesByFolder['root'].push(file);
                            }
                        });
                        
                        let html = '<div class="files-list"><strong>Fichiers .h5 trouvés:</strong><br>';
                        
                        // Fichiers à la racine
                        if (filesByFolder['root']) {
                            filesByFolder['root'].forEach(file => {
                                html += `<div style="margin-left: 10px;">📄 ${file}</div>`;
                            });
                        }
                        
                        // Fichiers dans les sous-dossiers
                        Object.keys(filesByFolder).forEach(folder => {
                            if (folder !== 'root') {
                                html += `<div style="margin-top: 10px; font-weight: bold;">📁 ${folder}/</div>`;
                                filesByFolder[folder].forEach(file => {
                                    html += `<div style="margin-left: 20px;">📄 ${file}</div>`;
                                });
                            }
                        });
                        
                        html += '</div>';
                        availableFilesDiv.innerHTML = html;
                        // availableFilesDiv.innerHTML = `<strong>Fichiers CDF trouvés:</strong><br>${data.files.join(', ')}`;
                        availableFilesDiv.style.display = 'block';
                        displayMessage(`${data.files.length} fichier(s) .h5 trouvé(s)`);
                    } else {
                        availableFilesDiv.innerHTML = '<strong>Aucun fichier .h5 trouvé dans ce dossier</strong>';
                        availableFilesDiv.style.display = 'block';
                        displayMessage('Aucun fichier .h5 trouvé', 'error');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                    availableFilesDiv.style.display = 'none';
                }
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listH5Btn.disabled = false;
                listH5Btn.textContent = '📋 Lister fichiers HDF5';
            }
        });        



        
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

            // ✅ Ouvrir JupyterLab dans le navigateur LOCAL si l'analyse réussit
              
                if (result.success && result.jupyter_url) {
                    window.open(result.jupyter_url, '_blank');
                    displayMessage('✅ Jupyter Lab ouvert dans un nouvel onglet', 'success');
                }



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
