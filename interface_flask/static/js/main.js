let selectedH5Files = [];

function fillDefaultPaths() {
    const form = document.getElementById('converterForm');
    const dockerPath = form.dataset.dockerPath; // récupère la valeur côté client
    document.getElementById('inputPath').value = dockerPath;
    document.getElementById('outputPath').value = dockerPath + 'output';
    document.getElementById('analysisPath').value = dockerPath + 'output';
    displayMessage('Chemins par défaut remplis', 'info');
}

function clearAllFields() {
            document.getElementById('converterForm').reset();
            document.getElementById('analysisForm').reset();
            document.getElementById('output').innerHTML = '';
            document.getElementById('availableFiles').style.display = 'none';
            displayMessage('Tous les champs ont été effacés', 'info');
        }
        
        function showProgress(show = true) {
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');
            
            if (show) {
                progressBar.style.display = 'block';
                let width = 0;
                const interval = setInterval(() => {
                    if (width >= 90) {
                        clearInterval(interval);
                    } else {
                        width += Math.random() * 10;
                        progressFill.style.width = width + '%';
                    }
                }, 200);
            } else {
                progressBar.style.display = 'none';
                progressFill.style.width = '100%';
                setTimeout(() => progressFill.style.width = '0%', 500);
            }
        }
        
        // Gestion des onglets
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            document.getElementById('output').innerHTML = '';
            showProgress(false);
        }
        
        const outputDiv = document.getElementById('output');
        const loadingDiv = document.getElementById('loading');
        
        function displayMessage(message, type = 'success') {
            const timestamp = new Date().toLocaleTimeString();
            let className, prefix;
            
            switch(type) {
                case 'error':
                    className = 'error';
                    prefix = '❌';
                    break;
                case 'info':
                    className = 'info';
                    prefix = 'ℹ️';
                    break;
                default:
                    className = 'success';
                    prefix = '✅';
            }
            
            outputDiv.innerHTML += `<span class="${className}">[${timestamp}] ${prefix} ${message}</span>\n`;
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }
        
        // CONVERSION TAB
        // const listFilesBtn = document.getElementById('listFilesBtn');
        // const availableFilesDiv = document.getElementById('availableFiles');
        // const converterForm = document.getElementById('converterForm');
        
        // listFilesBtn.addEventListener('click', async function() {
        //     const inputPath = document.getElementById('inputPath').value;
            
        //     if (!inputPath.trim()) {
        //         displayMessage('Veuillez spécifier un chemin d\'entrée', 'error');
        //         return;
        //     }
            
        //     listFilesBtn.disabled = true;
        //     listFilesBtn.innerHTML = '🔄 <span>Chargement...</span>';
        //     showProgress(true);
            
        //     try {
        //         const response = await fetch('/api/list_files', {
        //             method: 'POST',
        //             headers: { 'Content-Type': 'application/json' },
        //             body: JSON.stringify({ path: inputPath, extension: '.cdf' })
        //         });
                
        //         const data = await response.json();
                
        //         if (data.success) {
        //             if (data.files.length > 0) {
        //                 availableFilesDiv.innerHTML = `<strong>📁 ${data.files.length} fichiers CDF trouvés:</strong><br>`;
        //                 data.files.forEach(file => {
        //                     availableFilesDiv.innerHTML += `<div class="file-item">📄 ${file}</div>`;
        //                 });
        //                 availableFilesDiv.style.display = 'block';
        //                 displayMessage(`${data.files.length} fichier(s) CDF trouvé(s)`);
        //             } else {
        //                 availableFilesDiv.innerHTML = '<strong>⚠️ Aucun fichier CDF trouvé dans ce dossier</strong>';
        //                 availableFilesDiv.style.display = 'block';
        //                 displayMessage('Aucun fichier CDF trouvé', 'error');
        //             }
        //         } else {
        //             displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
        //             availableFilesDiv.style.display = 'none';
        //         }
        //     } catch (error) {
        //         displayMessage('Erreur de connexion: ' + error.message, 'error');
        //     } finally {
        //         listFilesBtn.disabled = false;
        //         listFilesBtn.innerHTML = '📋 <span>Lister les fichiers CDF</span>';
        //         showProgress(false);
        //     }
        // });
       
        // const cdfInput = document.getElementById('cdfFiles');
        // cdfInput.addEventListener('change', () => {
        //     const files = Array.from(cdfInput.files);
        //     console.log(files.map(f => f.name));
        // });
            const listFilesBtn = document.getElementById('listFilesBtn');
            const availableFilesDiv = document.getElementById('availableFiles');

            async function listFiles(path) {
                try {
                    const response = await fetch('/api/list_files', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path, extension: '.cdf' })
                    });
                    const data = await response.json();

                    if (data.success) {
                        availableFilesDiv.innerHTML = '';

                        // Affiche les dossiers
                        data.folders.forEach(f => {
                            const div = document.createElement('div');
                            div.className = 'file-item folder';
                            div.textContent = `📂 ${f.name}`;
                            div.onclick = () => {
                                document.getElementById('inputPath').value = f.path;
                                listFiles(f.path); // explore le dossier
                            };
                            availableFilesDiv.appendChild(div);
                        });

                        // Affiche les fichiers
                        data.files.forEach(f => {
                            const div = document.createElement('div');
                            div.className = 'file-item file';
                            div.textContent = `📄 ${f.name}`;
                            availableFilesDiv.appendChild(div);
                        });

                        availableFilesDiv.style.display = 'block';
                    } else {
                        displayMessage(data.message, 'error');
                        availableFilesDiv.innerHTML = '';
                    }
                } catch (err) {
                    displayMessage('Erreur de connexion: ' + err.message, 'error');
                }
            }




        converterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(converterForm);
            const data = {
                input_path: formData.get('inputPath'),
                output_path: formData.get('outputPath'),
                files: formData.get('files')
            };
            
            if (!data.input_path.trim()) {
                displayMessage('Veuillez spécifier un chemin d\'entrée', 'error');
                return;
            }
            
            if (!data.output_path.trim()) {
                displayMessage('Veuillez spécifier un chemin de sortie', 'error');
                return;
            }
            
            loadingDiv.style.display = 'block';
            outputDiv.innerHTML = '';
            showProgress(true);
            
            try {
                const response = await fetch('/api/convert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                result.messages.forEach(msg => {
                    const isError = msg.toLowerCase().includes('erreur');
                    displayMessage(msg, isError ? 'error' : 'success');
                });
                
                if (result.success && result.converted_files.length > 0) {
                    let filesHtml = '<div class="converted-files"><strong>✨ Fichiers convertis avec succès:</strong><br>';
                    result.converted_files.forEach(file => {
                        const filename = file.split('/').pop();
                        filesHtml += `<div class="file-list-item">📄 ${filename}</div>`;
                    });
                    filesHtml += '</div>';
                    outputDiv.innerHTML += filesHtml;
                }

                if (result.success) {
                    displayMessage(`✨ Conversion terminée avec succès! (${result.converted_files.length} fichier(s) converti(s))`);
                } else {
                    displayMessage('❌ La conversion a échoué', 'error');
                }
                
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                loadingDiv.style.display = 'none';
                showProgress(false);
            }
        });
        
        // ANALYSIS TAB
        const listH5Btn = document.getElementById('listH5Btn');
        const checkDockerBtn = document.getElementById('checkDockerBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const analysisForm = document.getElementById('analysisForm');
        const h5FilesSelect = document.getElementById('h5Files');
        const dockerStatusDiv = document.getElementById('dockerStatus');
        
        listH5Btn.addEventListener('click', async function() {
            const analysisPath = document.getElementById('analysisPath').value;
            
            if (!analysisPath.trim()) {
                displayMessage('Veuillez spécifier un chemin pour les fichiers .h5', 'error');
                return;
            }
            
            listH5Btn.disabled = true;
            listH5Btn.innerHTML = '🔄 <span>Chargement...</span>';
            showProgress(true);
            
            try {
                const response = await fetch('/api/list_files', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: analysisPath, extension: '.h5' })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    h5FilesSelect.innerHTML = '';
                    if (data.files.length > 0) {
                        data.files.forEach(file => {
                            const option = document.createElement('option');
                            option.value = file;
                            option.textContent = `📊 ${file}`;
                            h5FilesSelect.appendChild(option);
                        });
                        displayMessage(`${data.files.length} fichier(s) .h5 trouvé(s)`);
                    } else {
                        const option = document.createElement('option');
                        option.textContent = '⚠️ Aucun fichier .h5 trouvé';
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
                listH5Btn.innerHTML = '📋 <span>Lister fichiers HDF5</span>';
                showProgress(false);
            }
        });
        
        async function checkDockerStatus(retries = 10, delayMs = 3000) {
            await new Promise(r => setTimeout(r, 3000));
            for (let i = 0; i < retries; i++) {
                try {
                    const res = await fetch('/api/check_containers', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await res.json();
                    if (data.all_running) {
                        return data;
                    }
                } catch (err) {
                    console.log("Erreur fetch checkDockerStatus:", err);
                }
                await new Promise(r => setTimeout(r, delayMs));
            }
            return null;
        }

        checkDockerBtn.addEventListener('click', async function() {
            checkDockerBtn.disabled = true;
            checkDockerBtn.innerHTML = '🔄 <span>Lancement Docker...</span>';
            showProgress(true);
            
            try {
                let response = await fetch('/api/check_containers', { method: 'POST' });
                let data = await response.json();

                if (!data.all_running) {
                    displayMessage('🚀 Démarrage des conteneurs Docker...', 'info');
                    const response = await fetch('/api/start_containers', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    data = await checkDockerStatus();
                }
                
                if (data && data.all_running) {
                    dockerStatusDiv.className = 'docker-status docker-running';
                    dockerStatusDiv.innerHTML = `
                        <div class="status-indicator status-running"></div>
                        🟢 Tous les conteneurs Docker sont en cours d'exécution
                    `;
                    displayMessage('Conteneurs Docker: Tous en cours d\'exécution');
                } else {
                    dockerStatusDiv.className = 'docker-status docker-stopped';
                    dockerStatusDiv.innerHTML = `
                        <div class="status-indicator status-stopped"></div>
                        🔴 Certains conteneurs Docker ne sont pas en cours d'exécution
                    `;
                    displayMessage('Certains conteneurs Docker ne sont pas actifs', 'error');
                }
                
                if (data && data.status) {
                    data.status.forEach(status => displayMessage(status, 'info'));
                }
                
            } catch (error) {
                displayMessage('Erreur lors de la vérification Docker: ' + error.message, 'error');
                dockerStatusDiv.className = 'docker-status docker-stopped';
                dockerStatusDiv.innerHTML = `
                    <div class="status-indicator status-stopped"></div>
                    ❌ Erreur lors de la vérification Docker
                `;
            } finally {
                checkDockerBtn.disabled = false;
                checkDockerBtn.innerHTML = '🐳 <span>Lancer Docker</span>';
                showProgress(false);
            }
        });
        
        analysisForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const analysisPath = document.getElementById('analysisPath').value;
            const selectedFiles = Array.from(h5FilesSelect.selectedOptions).map(option => option.value);
            
            loadingDiv.style.display = 'block';
            outputDiv.innerHTML = '';
            showProgress(true);
            
            const data = {
                analysis_path: analysisPath,
                selected_files: selectedFiles
            };
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                result.messages.forEach(msg => {
                    const isError = msg.toLowerCase().includes('erreur');
                    displayMessage(msg, isError ? 'error' : 'success');
                });
                
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                loadingDiv.style.display = 'none';
                showProgress(false);
            }
        });
        
        // MONITORING TAB
        const refreshStatusBtn = document.getElementById('refreshStatusBtn');
        const viewLogsBtn = document.getElementById('viewLogsBtn');
        const systemStatus = document.getElementById('systemStatus');
        
        refreshStatusBtn.addEventListener('click', async function() {
            refreshStatusBtn.disabled = true;
            refreshStatusBtn.innerHTML = '🔄 <span>Actualisation...</span>';
            
            try {
                const response = await fetch('/api/check_containers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                
                systemStatus.innerHTML = '';
                
                if (data.all_running) {
                    systemStatus.innerHTML += `
                        <div class="docker-status docker-running">
                            <div class="status-indicator status-running"></div>
                            🟢 Système opérationnel - Tous les services actifs
                        </div>
                    `;
                } else {
                    systemStatus.innerHTML += `
                        <div class="docker-status docker-stopped">
                            <div class="status-indicator status-stopped"></div>
                            🔴 Certains services ne sont pas disponibles
                        </div>
                    `;
                }
                
                if (data.status) {
                    systemStatus.innerHTML += '<div style="margin-top: 16px;"><strong>📋 Détails des services:</strong></div>';
                    data.status.forEach(status => {
                        systemStatus.innerHTML += `
                            <div style="background: white; padding: 8px 12px; margin: 4px 0; border-radius: 6px; border-left: 4px solid #3b82f6;">
                                ${status}
                            </div>
                        `;
                    });
                }
                
                displayMessage('État du système actualisé', 'info');
                
            } catch (error) {
                systemStatus.innerHTML = `
                    <div class="docker-status docker-stopped">
                        <div class="status-indicator status-stopped"></div>
                        ❌ Impossible de vérifier l'état du système
                    </div>
                `;
                displayMessage('Erreur lors de l\'actualisation: ' + error.message, 'error');
            } finally {
                refreshStatusBtn.disabled = false;
                refreshStatusBtn.innerHTML = '🔄 <span>Actualiser l\'état</span>';
            }
        });
        
        viewLogsBtn.addEventListener('click', function() {
            displayMessage('📜 Affichage des logs système...', 'info');
            outputDiv.innerHTML += `
                <div style="border-top: 2px solid #374151; margin-top: 16px; padding-top: 16px;">
                    <strong>📊 Logs système (simulation):</strong><br>
                    [${new Date().toLocaleTimeString()}] INFO: Conteneur jupyter-lab démarré<br>
                    [${new Date().toLocaleTimeString()}] INFO: Port 8888 exposé avec succès<br>
                    [${new Date().toLocaleTimeString()}] INFO: Service prêt pour les connexions<br>
                    [${new Date().toLocaleTimeString()}] INFO: Dernière conversion: ${new Date().toLocaleDateString()}<br>
                </div>
            `;
            outputDiv.scrollTop = outputDiv.scrollHeight;
        });
        
        // Initialisation au chargement
        window.addEventListener('load', function() {
            // Vérification automatique de Docker
            fetch('/api/check_containers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            }).then(response => response.json())
            .then(data => {
                if (data.all_running) {
                    dockerStatusDiv.className = 'docker-status docker-running';
                    dockerStatusDiv.innerHTML = `
                        <div class="status-indicator status-running"></div>
                        🟢 Tous les conteneurs Docker sont en cours d'exécution
                    `;
                } else {
                    dockerStatusDiv.className = 'docker-status docker-stopped';
                    dockerStatusDiv.innerHTML = `
                        <div class="status-indicator status-stopped"></div>
                        🔴 Certains conteneurs Docker ne sont pas en cours d'exécution
                    `;
                }
            }).catch(() => {
                dockerStatusDiv.className = 'docker-status docker-stopped';
                dockerStatusDiv.innerHTML = `
                    <div class="status-indicator status-stopped"></div>
                    ❌ Impossible de vérifier l'état Docker
                `;
            });
            
            displayMessage('🚀 Interface Data Converter Pro initialisée', 'success');
        });
        
        // Détection de la perte de focus (fenêtre minimisée)
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                console.log('🗕 Fenêtre minimisée - Flask reste actif en arrière-plan');
            } else {
                displayMessage('👋 Interface de retour - Service Flask toujours actif', 'success');
            }
        });
        
        // Raccourcis clavier
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case '1':
                        e.preventDefault();
                        document.querySelector('.tab[onclick*="conversion"]').click();
                        break;
                    case '2':
                        e.preventDefault();
                        document.querySelector('.tab[onclick*="analysis"]').click();
                        break;
                    case '3':
                        e.preventDefault();
                        document.querySelector('.tab[onclick*="monitoring"]').click();
                        break;
                }
            }
        });