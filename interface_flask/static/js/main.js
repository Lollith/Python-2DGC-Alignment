let selectedH5Files = [];
let currentPath = '';
let targetInput = null;

const outputDiv = document.getElementById('output');

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
    const identificationPath = document.getElementById('identificationPath');
    const cleanupPath = document.getElementById('cleanupPath');
    
    if (inputPath) inputPath.value = dockerPath;
    if (outputPath) outputPath.value = dockerPath;
    if (analysisPath) analysisPath.value = dockerPath;
    if (identificationPath) identificationPath.value = dockerPath;
    if (cleanupPath) cleanupPath.value = dockerPath ;
    
    displayMessage('Chemins par défaut remplis', 'info');
}


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
    // if (output) output.innerHTML = '';
    
    showProgress(false);
}


function displayMessage(message, type = 'success') {
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

//------------explorateur de fichiers-------------------
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

function getParentPath(currentPath) {
    // Remplace backslash par slash pour compatibilité
    const safePath = currentPath.replace(/\\/g, '/').replace(/\/$/, '');
    if (safePath === '' || safePath === '/' || /^[A-Za-z]:[\/\\]?$/.test(safePath)) {
        // On est à la racine ("" ou "/" ou "C:/")
        return safePath;
    }
    // Gestion des chemins Windows type "C:/Users/..." ou Linux "/home/..."
    const parts = safePath.split('/');
    parts.pop();
    let parent = parts.join('/');
    if (parent === '') {
        // Retourne racine Windows "C:/" ou Linux "/"
        if (/^[A-Za-z]:/.test(safePath)) return safePath.substring(0, 2) + '/';
        return '/';
    }
    return parent;
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
            const parentPath = getParentPath(currentPath);
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

const loadingDiv = document.getElementById('loading');

// Initialisation principale
function initializeApp() {
    const outputDiv = document.getElementById('output');
    
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
            // outputDiv.innerHTML = '';
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
        // Event listeners pour le nettoyage des fichiers H5
    const listH5FilesBtn = document.getElementById('listH5FilesBtn');
    const deleteH5FilesBtn = document.getElementById('deleteH5FilesBtn');
    const h5FilesList = document.getElementById('h5FilesList');

    if (listH5FilesBtn) {
        listH5FilesBtn.addEventListener('click', async function() {
            const cleanupPath = document.getElementById('cleanupPath').value;
            if (!cleanupPath.trim()) {
                displayMessage('Veuillez spécifier un chemin pour le nettoyage', 'error');
                return;
            }

            listH5FilesBtn.disabled = true;
            listH5FilesBtn.textContent = '🔄 Recherche...';
            
            try {
                const response = await fetch('/api/list_files', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: cleanupPath, extension: '.h5' })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    if (data.files.length > 0) {
                        h5FilesList.innerHTML = `<strong>Fichiers .h5 trouvés (${data.files.length}):</strong><br>${data.files.join('<br>')}`;
                        h5FilesList.style.display = 'block';
                        displayMessage(`${data.files.length} fichier(s) .h5 trouvé(s)`, 'info');
                    } else {
                        h5FilesList.innerHTML = '<strong>Aucun fichier .h5 trouvé dans ce dossier</strong>';
                        h5FilesList.style.display = 'block';
                        displayMessage('Aucun fichier .h5 trouvé', 'info');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                    h5FilesList.style.display = 'none';
                }
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listH5FilesBtn.disabled = false;
                listH5FilesBtn.textContent = '📋 Lister les fichiers .h5';
            }
        });
    }

    if (deleteH5FilesBtn) {
        deleteH5FilesBtn.addEventListener('click', async function() {
            const cleanupPath = document.getElementById('cleanupPath').value;
            if (!cleanupPath.trim()) {
                displayMessage('Veuillez spécifier un chemin pour le nettoyage', 'error');
                return;
            }

            // Confirmation de suppression
            if (!confirm('⚠️ Êtes-vous sûr de vouloir supprimer TOUS les fichiers .h5 dans ce dossier ?\nCette action est irréversible !')) {
                return;
            }

            deleteH5FilesBtn.disabled = true;
            deleteH5FilesBtn.textContent = '🗑️ Suppression...';
            
            try {
                const response = await fetch('/api/delete_h5_files', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: cleanupPath })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayMessage(`✅ ${data.deleted_count} fichier(s) .h5 supprimé(s)`, 'success');
                    h5FilesList.style.display = 'none';
                } else {
                    displayMessage(data.message || 'Erreur lors de la suppression', 'error');
                }
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                deleteH5FilesBtn.disabled = false;
                deleteH5FilesBtn.textContent = '🗑️ Supprimer tous les .h5';
            }
        });
    }
    initializeAnalysisTab();
    displayMessage('🚀 Interface DataLab 2DGC initialisée', 'success');
    initializeIdentificationTab();
}

//----------Initialisation l'onglet Analysis-----------------
function initializeAnalysisTab() {
    const listH5Btn = document.getElementById('listH5Btn');
    const checkDockerBtn = document.getElementById('checkDockerBtn');
    const analysisForm = document.getElementById('analysisForm');
    const h5FilesSelect = document.getElementById('h5Files');
    const dockerStatusDiv = document.getElementById('dockerStatus');
    const viewLogsBtn = document.getElementById('viewLogsBtn');
    if (viewLogsBtn) {
        viewLogsBtn.addEventListener('click', viewLogs);
    }
    const refreshStatusBtn = document.getElementById('refreshStatusBtn');
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', refreshAllStatus);
    }

    const restartDockerBtn = document.getElementById('restartDockerBtn');
    if (restartDockerBtn) {
        restartDockerBtn.addEventListener('click', restartDocker);
    }

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

        // checkNistStatus(); // Vérification initiale au chargement
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
                
                // Vérifier NIST
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


    //----------onglet Identification-----------------
function initializeIdentificationTab() {
    const listCsvBtn = document.getElementById('listCsvBtn');
    const identificationForm = document.getElementById('identificationForm');
    const csvFilesSelect = document.getElementById('csvFiles');


    listCsvBtn.addEventListener('click', async function() {
            const identificationPath = document.getElementById('identificationPath').value;
            
            if (!identificationPath.trim()) {
                displayMessage('Veuillez spécifier un chemin pour les fichiers .csv', 'error');
                return;
            }
            listCsvBtn.disabled = true;
            listCsvBtn.textContent = '🔄 Chargement...';
            
            try {
                const response = await fetch('/api/list_files', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ path: identificationPath, extension: '.csv' })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    csvFilesSelect.innerHTML = '';
                    if (data.files.length > 0) {
                        data.files.forEach(file => {
                            const option = document.createElement('option');
                            option.value = file;
                            option.textContent = file;
                            csvFilesSelect.appendChild(option);
                        });
                        displayMessage(`${data.files.length} fichier(s) .csv trouvé(s)`);
                    } else {
                        const option = document.createElement('option');
                        option.textContent = 'Aucun fichier .csv trouvé';
                        option.disabled = true;
                        csvFilesSelect.appendChild(option);
                        displayMessage('Aucun fichier .csv trouvé', 'error');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                }
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listCsvBtn.disabled = false;
                listCsvBtn.textContent = '📋 Lister fichiers CSV';
            }
        });
   

// Lancer l'analyse
        identificationForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const identificationPath = document.getElementById('identificationPath').value;
            const selectedFiles = Array.from(csvFilesSelect.selectedOptions).map(option => option.value);
            loadingDiv.style.display = 'block';

            
            const data = {
                identification_path: identificationPath,
                selected_files: selectedFiles
            };

            //TODO check nist

            //TODO changer ici

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
    }


       





// // ---------------------Onglet Monitoring--------------------
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
                const isSuccess = msg.includes('✅');
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


// récupérer les logs via l'API
async function viewLogs() {
    const viewLogsBtn = document.getElementById('viewLogsBtn');
    if (!viewLogsBtn) return;
    
    viewLogsBtn.disabled = true;
    viewLogsBtn.innerHTML = '📜 <span>Chargement logs...</span>';
    
    try {
        const response = await fetch('/api/logs', { method: 'GET' });
        const data = await response.json();
        
        if (outputDiv) {
            // outputDiv.innerHTML = '';
            displayMessage('=== LOGS SYSTÈME ===', 'info');
            
            if (data.success) {
                data.logs.forEach(log => {
                    const isError = log.includes('❌') || log.includes('Erreur');
                    const isSuccess = log.includes('✅') || log.includes('🟢');
                    const type = isError ? 'error' : (isSuccess ? 'success' : 'info');
                    displayMessage(log, type);
                });
            } else {
                displayMessage('Erreur lors de la récupération des logs', 'error');
            }
            
            displayMessage('=== FIN DES LOGS ===', 'info');
        }
        
    } catch (error) {
        if (outputDiv) {
            // outputDiv.innerHTML = '';
            displayMessage('❌ Erreur de connexion pour récupérer les logs', 'error');
        }
    } finally {
        viewLogsBtn.disabled = false;
        viewLogsBtn.innerHTML = '📜 <span>Voir les logs</span>';
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);

document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('🗕 Fenêtre minimisée - Flask reste actif en arrière-plan');
    } else {
        displayMessage('👋 Interface de retour - Service Flask toujours actif', 'success');
    }
});

//------------Raccourcis clavier-------------------
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