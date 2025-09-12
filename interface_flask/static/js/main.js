let selectedH5Files = [];
let currentPath = '';
let targetInput = null;

function fillDefaultPaths() {
    const form = document.getElementById('dockerPathMeta');
    if (!form) {
        console.error('Formulaire dockerPathMeta non trouvé');
        return;
    }
    
    const dockerPath = form.dataset.dockerPath;
    const inputPath = document.getElementById('inputPath');
    const outputPath = document.getElementById('outputPath');
    const analysisPath = document.getElementById('analysisPath');
    
    if (inputPath) inputPath.value = dockerPath;
    if (outputPath) outputPath.value = dockerPath ;
    if (analysisPath) analysisPath.value = dockerPath ;
    
    displayMessage('Chemins par défaut remplis', 'info');
}

// function clearAllFields() {
//     const converterForm = document.getElementById('converterForm');
//     const analysisForm = document.getElementById('analysisForm');
//     const output = document.getElementById('output');
//     const availableFiles = document.getElementById('availableFiles');
    
//     if (converterForm) converterForm.reset();
//     if (analysisForm) analysisForm.reset();
//     if (output) output.innerHTML = '';
//     if (availableFiles) availableFiles.style.display = 'none';
    
//     displayMessage('Tous les champs ont été effacés', 'info');
// }

function showProgress(show = true) {
    const progressBar = document.getElementById('progressBar');
    const progressFill = document.getElementById('progressFill');
    
    if (!progressBar || !progressFill) return;
    
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

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    const tabContent = document.getElementById(tabName);
    const output = document.getElementById('output');
    
    if (tabContent) tabContent.classList.add('active');
    if (event && event.target) event.target.classList.add('active');
    if (output) output.innerHTML = '';
    
    showProgress(false);
}

function displayMessage(message, type = 'success') {
    const outputDiv = document.getElementById('output');
    if (!outputDiv) return;
    
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




// Fonction pour l'explorateur de fichiers
function openFileExplorer(inputId) {
    targetInput = document.getElementById(inputId);
    if (!targetInput) {
        console.error(`Input ${inputId} non trouvé`);
        return;
    }
    
    currentPath = targetInput.value || '/';
    loadDirectoryContent(currentPath);
    
    const modalElement = document.getElementById('fileExplorerModal');
    if (modalElement && typeof bootstrap !== 'undefined') {
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    } else {
        displayMessage('Modal d\'exploration non disponible', 'error');
    }
}

async function loadDirectoryContent(path) {
    try {
        const response = await fetch('/api/browse_files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path })
        });
        
        const data = await response.json();
        console.log('Réponse de /api/browse_files:', data); // AJOUTE CE LOG

        currentPath = path;
        
        if (data.success) {
            displayFileList(data.folders || [], data.files || [], path);
            const currentPathElement = document.getElementById('currentPath');
            if (currentPathElement) {
                currentPathElement.textContent = `Dossier: ${path}`;
            }
        } else {
            displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
        }
    } catch (error) {
        console.error('Erreur:', error);
        displayMessage('Erreur de connexion: ' + error.message, 'error');
    }
}

function displayFileList(folders, files, currentPath) {
    const fileList = document.getElementById('fileList');
    if (!fileList) return;
    
    fileList.innerHTML = '';
    
    // Bouton Parent Directory
    if (currentPath !== '/') {
        const parentDiv = document.createElement('div');
        parentDiv.className = 'file-item folder';
        parentDiv.innerHTML = '📁 .. (Dossier parent)';
        parentDiv.onclick = () => {
            const parentPath = currentPath.split('/').slice(0, -1).join('/') || '/';
            loadDirectoryContent(parentPath);
        };
        fileList.appendChild(parentDiv);
    }
    
    // Dossiers
    folders.forEach(folder => {
        const div = document.createElement('div');
        div.className = 'file-item folder';
        div.innerHTML = `📂 ${folder.name}`;
        div.onclick = () => {
            loadDirectoryContent(folder.path);
        };
        fileList.appendChild(div);
    });
    
    // Fichiers
    files.forEach(file => {
        const div = document.createElement('div');
        div.className = 'file-item file';
        div.innerHTML = `📄 ${file.name}`;
        fileList.appendChild(div);
    });
}

// Fonction d'initialisation principale
function initializeApp() {
    const outputDiv = document.getElementById('output');
    const loadingDiv = document.getElementById('loading');
    
    if (!outputDiv || !loadingDiv) {
        console.error('Éléments de base manquants dans le DOM');
        return;
    }

    // Initialisation du bouton de sélection de dossier
    const selectFolderBtn = document.getElementById('selectFolder');
    if (selectFolderBtn) {
        selectFolderBtn.onclick = () => {
            if (targetInput) {
                targetInput.value = currentPath;
                const modalElement = document.getElementById('fileExplorerModal');
                if (modalElement && typeof bootstrap !== 'undefined') {
                    bootstrap.Modal.getInstance(modalElement)?.hide();
                }
                displayMessage(`Dossier sélectionné: ${currentPath}`, 'info');
            }
        };
    }



    const listFilesBtn = document.getElementById('listFilesBtn');
    const availableFilesDiv = document.getElementById('availableFiles');
    // Lister les fichiers CDF
    listFilesBtn.addEventListener('click', async function() {
        const inputPath = document.getElementById('inputPath').value;
        
        if (!inputPath.trim()) {
            displayMessage('Veuillez spécifier un chemin d\'entrée', 'error');
            return;
        }
        
        listFilesBtn.disabled = true;
        listFilesBtn.textContent = '🔄 Chargement...';
        
        try {
            const response = await fetch('/api/list_files', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: inputPath, extension: '.cdf' })
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (data.files.length > 0) {
                    availableFilesDiv.innerHTML = `<strong>Fichiers CDF trouvés:</strong><br>${data.files.join(', ')}`;
                    availableFilesDiv.style.display = 'block';
                    displayMessage(`${data.files.length} fichier(s) CDF trouvé(s)`);
                } else {
                    availableFilesDiv.innerHTML = '<strong>Aucun fichier CDF trouvé dans ce dossier</strong>';
                    availableFilesDiv.style.display = 'block';
                    displayMessage('Aucun fichier CDF trouvé', 'error');
                }
            } else {
                displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                availableFilesDiv.style.display = 'none';
            }
        } catch (error) {
            displayMessage('Erreur de connexion: ' + error.message, 'error');
        } finally {
            listFilesBtn.disabled = false;
            listFilesBtn.textContent = '📋 Lister les fichiers CDF';
        }
    });

    
    // Event listener pour le formulaire de conversion
    const converterForm = document.getElementById('converterForm');
    if (converterForm) {
        converterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(converterForm);
            const data = {
                input_path: formData.get('inputPath'),
                output_path: formData.get('outputPath'),
                files: formData.get('files')
            };
            
            if (!data.input_path?.trim()) {
                displayMessage('Veuillez spécifier un chemin d\'entrée', 'error');
                return;
            }
            
            if (!data.output_path?.trim()) {
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
                
                if (result.messages) {
                    result.messages.forEach(msg => {
                        const isError = msg.toLowerCase().includes('erreur');
                        displayMessage(msg, isError ? 'error' : 'success');
                    });
                }
                
                if (result.success && result.converted_files?.length > 0) {
                    let filesHtml = '<div class="converted-files"><strong>✨ Fichiers convertis avec succès:</strong><br>';
                    result.converted_files.forEach(file => {
                        const filename = file.split('/').pop();
                        filesHtml += `<div class="file-list-item">📄 ${filename}</div>`;
                    });
                    filesHtml += '</div>';
                    outputDiv.innerHTML += filesHtml;
                }

                if (result.success) {
                    displayMessage(`✨ Conversion terminée avec succès! (${result.converted_files?.length || 0} fichier(s) converti(s))`);
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
    }

    // Initialisation des autres fonctionnalités...
    initializeAnalysisTab();
    initializeMonitoringTab();
    initializeDockerStatus();

    displayMessage('🚀 Interface DataLab 2DGC initialisée', 'success');
}

// Fonction pour initialiser l'onglet Analysis
function initializeAnalysisTab() {
    const listH5Btn = document.getElementById('listH5Btn');
    const checkDockerBtn = document.getElementById('checkDockerBtn');
    const analysisForm = document.getElementById('analysisForm');
    const h5FilesSelect = document.getElementById('h5Files');
    const dockerStatusDiv = document.getElementById('dockerStatus');
    const analyzeBtn = document.getElementById('analyzeBtn');


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
        
         async function checkDockerStatus(retries = 10, delayMs = 100000) {
            await new Promise(r => setTimeout(r, 100000));
            for (let i = 0; i < retries; i++) {
                try {
                    const res = await fetch('/api/check_containers', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await res.json();
                    if (data.all_running) {
                        return data; // Docker est up, on retourne les données
                    }
                } catch (err) {
                    console.log("Erreur fetch checkDockerStatus:", err);
                }
                await new Promise(r => setTimeout(r, delayMs));
            }
            return null; // Docker n'a pas démarré après toutes les tentatives
        }


        // Vérifier l'état des conteneurs Docker
        checkDockerBtn.addEventListener('click', async function() {
            checkDockerBtn.disabled = true;
            checkDockerBtn.textContent = '🔄 Lancement Docker...';
            
            try {
                // 1️⃣ Vérifier si Docker est déjà lancé
                let response = await fetch('/api/check_containers', { method: 'POST' });
                let data = await response.json();

                 if (!data.all_running) {
                   // 2️⃣ Lancer Docker seulement si nécessaire

                    const response = await fetch('/api/start_containers', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                // const data = await response.json();
                const data = await checkDockerStatus();
                }
                
                if (data && data.all_running) {
                    dockerStatusDiv.className = 'docker-status docker-running';
                    dockerStatusDiv.innerHTML = '🟢 Tous les conteneurs Docker sont en cours d\'exécution';
                    displayMessage('Conteneurs Docker: Tous en cours d\'exécution');
                } else {
                    dockerStatusDiv.className = 'docker-status docker-stopped';
                    dockerStatusDiv.innerHTML = '🔴 Certains conteneurs Docker ne sont pas en cours d\'exécution';
                    displayMessage('Certains conteneurs Docker ne sont pas actifs', 'error');
                }
                
                // Afficher les détails
                // data.status.forEach(status => {
                //     displayMessage(status, 'info');
                // });
                if (data && data.status) {
                    data.status.forEach(status => displayMessage(status, 'info'));
                }
                
            } catch (error) {
                displayMessage('Erreur lors de la vérification Docker: ' + error.message, 'error');
                dockerStatusDiv.className = 'docker-status docker-stopped';
                dockerStatusDiv.innerHTML = '❌ Erreur lors de la vérification Docker';
            } finally {
                checkDockerBtn.disabled = false;
                checkDockerBtn.textContent = '🐳 Vérifier Docker';
            }
        });
// Lancer l'analyse
        analysisForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const analysisPath = document.getElementById('analysisPath').value;
            const selectedFiles = Array.from(h5FilesSelect.selectedOptions).map(option => option.value);
            
            // Validation
            // if (!analysisPath.trim()) {
            //     displayMessage('Veuillez spécifier un chemin pour les fichiers .npy', 'error');
            //     return;
            // }
            
            // if (selectedFiles.length === 0) {
            //     displayMessage('Veuillez sélectionner au moins un fichier .npy à analyser', 'error');
            //     return;
            // }
            
            // Afficher le chargement
            loadingDiv.style.display = 'block';
            outputDiv.innerHTML = '';
            
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
                
                // if (result.success) {
                //     displayMessage(`✨ Analyse terminée avec succès!`);
                // } else {
                //     displayMessage('❌ L\'analyse a échoué', 'error');
                // }
                
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
                    dockerStatusDiv.className = 'docker-status docker-running';
                    dockerStatusDiv.innerHTML = '🟢 Tous les conteneurs Docker sont en cours d\'exécution';
                } else {
                    dockerStatusDiv.className = 'docker-status docker-stopped';
                    dockerStatusDiv.innerHTML = '🔴 Certains conteneurs Docker ne sont pas en cours d\'exécution';
                }
            }).catch(() => {
                dockerStatusDiv.className = 'docker-status docker-stopped';
                dockerStatusDiv.innerHTML = '❌ Impossible de vérifier l\'état Docker';
            });
        });
    
    
    }






// Fonction pour initialiser l'onglet Monitoring
function initializeMonitoringTab() {
    const refreshStatusBtn = document.getElementById('refreshStatusBtn');
    const viewLogsBtn = document.getElementById('viewLogsBtn');

    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', async function() {
            // Votre code existant pour refresh
        });
    }

    if (viewLogsBtn) {
        viewLogsBtn.addEventListener('click', function() {
            // Votre code existant pour les logs
        });
    }
}

// Initialisation automatique du statut Docker
function initializeDockerStatus() {
    const dockerStatusDiv = document.getElementById('dockerStatus');
    if (!dockerStatusDiv) return;

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
}

// Event listeners globaux
document.addEventListener('DOMContentLoaded', initializeApp);

document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('🗕 Fenêtre minimisée - Flask reste actif en arrière-plan');
    } else {
        displayMessage('👋 Interface de retour - Service Flask toujours actif', 'success');
    }
});

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case '1':
                e.preventDefault();
                const tab1 = document.querySelector('.tab[onclick*="conversion"]');
                if (tab1) tab1.click();
                break;
            case '2':
                e.preventDefault();
                const tab2 = document.querySelector('.tab[onclick*="analysis"]');
                if (tab2) tab2.click();
                break;
            case '3':
                e.preventDefault();
                const tab3 = document.querySelector('.tab[onclick*="monitoring"]');
                if (tab3) tab3.click();
                break;
        }
    }
});