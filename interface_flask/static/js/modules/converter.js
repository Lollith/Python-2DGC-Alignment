import { displayMessage, loadingDiv, outputDiv, getTargetInput, getCurrentPath, showProgress } from '../main.js';

/**
 * Initialize the converter tab: sets up folder selection, lists CDF files,
 * runs the conversion process, and manages HDF5 cleanup utilities.
 */

export function initializeConverterTab() {
    const listFilesBtn = document.getElementById('listFilesBtn');
    const availableFilesDiv = document.getElementById('availableFiles');
    const converterForm = document.getElementById('converterForm');
    
    // Initialisation du bouton de sélection de dossier
    const selectFolderBtn = document.getElementById('selectFolder');
    if (selectFolderBtn) {
        selectFolderBtn.onclick = () => {
            const targetInput = getTargetInput();
            const currentPath = getCurrentPath();

            if (targetInput) {
                targetInput.value = currentPath;
                const modalElement = document.getElementById('fileExplorerModal');
                if (modalElement && typeof bootstrap !== 'undefined') {
                    bootstrap.Modal.getInstance(modalElement)?.hide();
                }
            }
        };
    }

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
                if (data.messages && data.messages.length > 0) {
                    data.messages.forEach(msg => {
                    const isWarning = msg.includes('⚠️') || msg.includes('ATTENTION');
                    displayMessage(msg, isWarning ? 'warning' : 'info');
                    });
                }

                if (data.files.length > 0) {
                    // Affichage groupé par dossier
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
                    
                    let html = '<div class="files-list"><strong>Fichiers CDF trouvés:</strong><br>';
                    
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
                    if (data.messages && data.messages.length > 0) {
                        data.messages.forEach(msg => {
                            displayMessage(msg, msg.includes('⚠️') || msg.includes('ATTENTION') ? 'warning' : 'info');
                        });
                    }
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
                    displayMessage(`✨ ${data.deleted_count} fichier(s) .h5 supprimé(s)`, 'success');
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
    displayMessage('🚀 Interface DataLab 2DGC initialisée', 'success');
}