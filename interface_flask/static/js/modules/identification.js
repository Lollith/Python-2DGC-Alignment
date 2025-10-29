import { displayMessage, loadingDiv, showProgress } from '../main.js';

export function initializeIdentificationTab() {
    const listCsvBtn = document.getElementById('listCsvBtn');
    const csvFilesInput = document.getElementById('csvFiles');
    const availableCsvFilesDiv = document.getElementById('availableCsvFiles');
    const identificationForm = document.getElementById('identificationForm');


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
                    body: JSON.stringify({ path: identInputPath, extension: '.csv' })
                });
                
                const data = await response.json();
                if (data.success) {
                    if (data.files.length > 0) {
                        availableCsvFilesDiv.innerHTML = `<strong>Fichiers CSV trouvés:</strong><br>${data.files.join(', ')}`;
                        availableCsvFilesDiv.style.display = 'block';
                        displayMessage(`${data.files.length} fichier(s) CSV trouvé(s)`);
                    } else {
                        availableCsvFilesDiv.innerHTML = '<strong>Aucun fichier CSV trouvé dans ce dossier</strong>';
                        availableCsvFilesDiv.style.display = 'block';
                        displayMessage('Aucun fichier CSV trouvé', 'error');
                    }
                } else {
                    displayMessage(data.message || 'Erreur lors de la lecture du dossier', 'error');
                    availableCsvFilesDiv.style.display = 'none';
                }
                
            } catch (error) {
                displayMessage('Erreur de connexion: ' + error.message, 'error');
            } finally {
                listCsvBtn.disabled = false;
                listCsvBtn.textContent = '📋 Lister fichiers CSV';
            }
        });
   

// Lancer l'analyse
    
    if (identificationForm) {
        identificationForm.addEventListener('submit', async function(e) {
            e.preventDefault();
        
            const formData = new FormData(identificationForm);
            const data = {
                input_path: formData.get('identInputPath'),
                output_path: formData.get('identOutputPath'),
                files: formData.get('csvFiles')
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
            const nistCheck = await fetch('/nist/health');
            displayMessage('Vérification du statut du moteur NIST...');
            const nistStatus = await nistCheck.json();
            
            if (nistStatus.nist_status !== 'available') {
                displayMessage('❌ Moteur NIST indisponible. Vérifiez le statut dans l\'onglet Monitoring.', 'error');
                return;
            } else {
                displayMessage('✅ Moteur NIST actif.');
            }
            } catch (error) {
                displayMessage('❌ Impossible de vérifier le statut NIST: ' + error.message, 'error');
                return;
            }

            try {
            displayMessage('🔬 Lancement de l\'identification NIST...', 'info');
            
            const response = await fetch('/api/identify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                displayMessage(`✅ Identification terminée! ${result.identified_compounds || 0} composé(s) identifié(s)`, 'success');
                
                // Afficher les résultats
                if (result.results?.length > 0) {
                    let resultsHtml = '<div class="identification-results"><strong>🔬 Résultats d\'identification:</strong><br>';
                    result.results.forEach(compound => {
                        resultsHtml += `<div class="result-item">📄 ${compound.name} (Score: ${compound.score})</div>`;
                    });
                    resultsHtml += '</div>';
                    outputDiv.innerHTML += resultsHtml;
                    displayMessage('✅ Résultats affichés ci-dessous.', result.message);
                }
            } else {
                displayMessage('❌ L\'identification a échoué: ' + result.message, 'error');
            }
            
        } catch (error) {
            displayMessage('❌ Erreur de connexion: ' + error.message, 'error');
        } finally {
            loadingDiv.style.display = 'none';
            showProgress(false);
        }
    });
}
}
       