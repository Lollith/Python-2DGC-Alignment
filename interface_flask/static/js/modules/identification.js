import { displayMessage, loadingDiv, outputDiv, showProgress } from '../main.js';

export function initializeIdentificationTab() {
    
    const listCsvBtn = document.getElementById('listCsvBtn');
    const availableCsvFilesDiv = document.getElementById('availableCsvFiles');
    const identificationForm = document.getElementById('identificationForm');

    if (identificationForm) {
        listCsvBtn.addEventListener('click', async function() {
            const identInputPath = document.getElementById('identInputPath').value;

            if (!identInputPath.trim()) {
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
                    body: JSON.stringify({
                        path: identInputPath,
                        extension: '.csv',
                        peak_info_only: true
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    if (data.files.length > 0) {
                        availableCsvFilesDiv.innerHTML = `<strong>Fichiers Peak_Info.csv trouvés:</strong><br>${data.files.join(', ')}`;
                        availableCsvFilesDiv.style.display = 'block';
                        displayMessage(`${data.files.length} fichier(s) Peak_Info.csv trouvé(s)`);
                    } else {
                        availableCsvFilesDiv.innerHTML = '<strong>Aucun fichier Peak_Info.csv trouvé dans ce dossier</strong>';
                        availableCsvFilesDiv.style.display = 'block';
                        displayMessage('Aucun fichier Peak_Info.csv trouvé', 'error');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                    availableCsvFilesDiv.style.display = 'none';
                }
                
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listCsvBtn.disabled = false;
                listCsvBtn.textContent = '📋 Lister fichiers';
            }
        });
    }
// Lancer l'analyse
    if (identificationForm) {
        let currentEventSource = null;

        identificationForm.addEventListener('submit', async function(e) {
            e.preventDefault();
             if (currentEventSource) {
                console.log('🔄 Fermeture ancienne EventSource');
                currentEventSource.close();
                currentEventSource = null;
            }

            const formData = new FormData(identificationForm);
            const data = {
                input_path: formData.get('identInputPath'),
                output_path: formData.get('identOutputPath'),
                files: formData.get('csvFiles'),
                match_factor_min: formData.get('matchFactorMin')
            };
            
            if (!data.input_path?.trim()) {
                displayMessage('Veuillez spécifier un chemin d\'entrée', 'error');
                return;
            }
            if (!data.output_path?.trim()) {
                displayMessage('Veuillez spécifier un chemin de sortie', 'error');
                return;
            }

            function cleanupUI() {
                loadingDiv.style.display = 'none';
                showProgress(false);
            }

            try {
                const nistCheck = await fetch('/nist/health');
                displayMessage('Vérification du statut du moteur NIST...');
                const nistStatus = await nistCheck.json();
            
                if (nistStatus.nist_status !== 'available') {
                    displayMessage('❌ Moteur NIST indisponible. Vérifiez le statut dans l\'onglet Monitoring.', 'error');
                    cleanupUI();
                    return;
                } else {
                    displayMessage('⚡ Moteur NIST actif.');
                }
            } catch (error) {
                displayMessage('❌ Impossible de vérifier le statut NIST: ' + error.message, 'error');
                cleanupUI();
                return;
            }
            
        // UI loading
            loadingDiv.style.display = 'block';
            showProgress(true);
            
            const params = new URLSearchParams(data);
            currentEventSource = new EventSource(`/api/identify?${params}`);

            currentEventSource.onopen = function(event) {
            };

            currentEventSource.onmessage = function(event) {
                try {
                    const messageData = JSON.parse(event.data);
                    switch(messageData.type) {
                        case 'message':
                            displayMessage(messageData.content, messageData.message_type);
                            break;
                        case 'error':
                            displayMessage(messageData.content, messageData.message_type);
                            currentEventSource.close();
                            loadingDiv.style.display = 'none';
                            showProgress(false);
                            break;
                        case 'complete':
                            displayMessage(messageData.content, messageData.message_type);
                            currentEventSource.close();
                            loadingDiv.style.display = 'none';
                            showProgress(false);
                            break;

                        default:
                            console.warn('⚠️ Message type inconnu:', messageData.type);
                            displayMessage(messageData.content || 'Message reçu', 'info');
                    }
                } catch (e) {
                    console.error('❌ Erreur parsing JSON:', e);
                    console.error('❌ Data reçue:', event.data);
                }
            };

            currentEventSource.onerror = function(event) {
                console.error('❌ EventSource error:', event);
                displayMessage('❌ Erreur de connexion au stream', 'error');
                currentEventSource.close();
                cleanupUI();
            };

            // Nettoyer si l'utilisateur quitte la page
            window.addEventListener('beforeunload', function() {
                eventSource.close();
            });
        });
    }
}
    