import { openFileExplorer, initializeFileExplorer } from './modules/fileExplorer.js';
import { initializeConverterTab } from './modules/converter.js';
import { initializeAnalysisTab } from './modules/analysis.js';
import { initializeIdentificationTab } from './modules/identification.js';
import { initializeMonitoringTab } from './modules/monitoring.js';

/**
 * Entry point for the DataLab 2DGC web interface.
 * Wires global helpers, initializes all feature tabs (converter, analysis,
 * identification, monitoring, file explorer), fills default paths, and
 * sets up UI utilities such as progress bar, messaging, keyboard shortcuts,
 * tab switching, and log viewing.
 */

export let selectedH5Files = [];
export let currentPath = '';
export let targetInput = null;

export function setCurrentPath(path) {
    currentPath = path;
}
export function setTargetInput(input) {
    targetInput = input;
}
export function getCurrentPath() {
    return currentPath;
}
export function getTargetInput() {
    return targetInput;
}

const outputDiv = document.getElementById('output');
const loadingDiv = document.getElementById('loading');
export {loadingDiv, outputDiv};

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
    const identInputPath = document.getElementById('identInputPath');
    const identOutputPath = document.getElementById('identOutputPath');
    const cleanupPath = document.getElementById('cleanupPath');
    
    if (inputPath) inputPath.value = dockerPath;
    if (analysisPath) analysisPath.value = dockerPath;
    if (identInputPath) identInputPath.value = dockerPath;
    if (cleanupPath) cleanupPath.value = dockerPath ;
   
    if (inputPath.value) {
        const inputPathValue = inputPath.value.trim();
        const normalizedPath = inputPathValue.replace(/\\/g, '/');
        const defaultOutput = normalizedPath.endsWith('/') 
            ? normalizedPath + 'output' 
            : normalizedPath + '/output';
        outputPath.value = defaultOutput;
    }
    if (identInputPath.value) {
        const identInputPathValue = identInputPath.value.trim();
        const normalizedIdentInputPath = identInputPathValue.replace(/\\/g, '/');
        const defaultOutput = normalizedIdentInputPath.endsWith('/') 
            ? normalizedIdentInputPath + 'output' 
            : normalizedIdentInputPath + '/output';
        identOutputPath.value = defaultOutput;
    }

    if (outputPath && !outputPath.dataset.listenerAdded) {
        outputPath.addEventListener('input', function() {
            this.dataset.userModified = 'true';
        });
        outputPath.dataset.listenerAdded = 'true';
    }
     if (identOutputPath && !identOutputPath.dataset.listenerAdded) {
        identOutputPath.addEventListener('input', function() {
            this.dataset.userModified = 'true';
        });
        identOutputPath.dataset.listenerAdded = 'true';
    }
    //converter
    if (inputPath && outputPath && !inputPath.dataset.listenerAdded) {
        inputPath.addEventListener('input', function() {
            const inputValue = this.value.trim();
            // Seulement si pas modifié manuellement par l'utilisateur
            if (inputValue && !outputPath.dataset.userModified) {
                const normalizedPath = inputValue.replace(/\\/g, '/');
                const defaultOutput = normalizedPath.endsWith('/') 
                    ? normalizedPath + 'output' 
                    : normalizedPath + '/output';
                outputPath.value = defaultOutput;
            }
        });
        inputPath.dataset.listenerAdded = 'true';
    }

    //identification
    if (identInputPath && identOutputPath && !identInputPath.dataset.listenerAdded) {
        identInputPath.addEventListener('input', function() {
            const inputValue = this.value.trim();
            // Seulement si pas modifié manuellement par l'utilisateur
            if (inputValue && !identOutputPath.dataset.userModified) {
                const normalizedPath = inputValue.replace(/\\/g, '/');
                const defaultOutput = normalizedPath.endsWith('/') 
                    ? normalizedPath + 'output' 
                    : normalizedPath + '/output';
                identOutputPath.value = defaultOutput;
            }
        });
        identInputPath.dataset.listenerAdded = 'true';
    }
    displayMessage('Chemins par défaut remplis', 'info');
}

export function showProgress(show = true) {
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
    
    if (tabContent) tabContent.classList.add('active');
    if (event && event.target) event.target.classList.add('active');
    
    showProgress(false);
}


export function displayMessage(message, type = 'success') {
    if (!outputDiv) return;
    
    const timestamp = new Date().toLocaleTimeString();
    let className, prefix;
    
    switch(type) {
        case 'error':
            className = 'error';
            prefix = '⚠️';
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

// récupérer les logs via l'API
export async function viewLogs() {
    const viewLogsBtn = document.getElementById('viewLogsBtn');
    if (!viewLogsBtn) return;
    
    viewLogsBtn.disabled = true;
    viewLogsBtn.innerHTML = '📜 <span>Chargement logs...</span>';
    
    try {
        const response = await fetch('/api/logs', { method: 'GET' });
        const data = await response.json();
        
        if (outputDiv) {
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
            displayMessage('❌ Erreur de connexion pour récupérer les logs', 'error');
        }
    } finally {
        viewLogsBtn.disabled = false;
        viewLogsBtn.innerHTML = '📜 <span>Voir les logs</span>';
    }
}

window.fillDefaultPaths = fillDefaultPaths;
window.showTab = showTab;
window.openFileExplorer = openFileExplorer;
window.getCurrentPath = getCurrentPath;
window.getTargetInput = getTargetInput;
window.setCurrentPath = setCurrentPath;
window.setTargetInput = setTargetInput;

document.addEventListener('DOMContentLoaded', async function() {
    // Rendre les fonctions globales pour les onclick HTML
    window.fillDefaultPaths = fillDefaultPaths;
    window.showTab = showTab;
    window.openFileExplorer = openFileExplorer;
    
    initializeFileExplorer();
    initializeConverterTab();
    initializeAnalysisTab();
    initializeIdentificationTab();
    initializeMonitoringTab();

    fillDefaultPaths();
    displayMessage('✨ Application initialisée', 'success');
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


document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('🗕 Fenêtre minimisée - Flask reste actif en arrière-plan');
    } else {
        displayMessage('👋 Interface de retour - Service Flask toujours actif', 'success');
    }
});

// Fix pour les modals Bootstrap aria-hidden
document.querySelectorAll('.modal').forEach((modal) => {
    modal.addEventListener('hide.bs.modal', () => {
        if (document.activeElement) {
            document.activeElement.blur();
        }
    });
});
