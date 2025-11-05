import { displayMessage, setCurrentPath, setTargetInput, getCurrentPath, getTargetInput } from '../main.js';

export function openFileExplorer(inputId) {
    const input = document.getElementById(inputId);
    if (!input) {
        console.error(`Input ${inputId} non trouvé`);
        return;
    }
    
    setTargetInput(input);
    setCurrentPath(input.value || '/');

    loadDirectoryContent(getCurrentPath());
    
    const modalElement = document.getElementById('fileExplorerModal');
    if (modalElement && typeof bootstrap !== 'undefined') {
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    } else {
        displayMessage('Modal d\'exploration non disponible', 'error');
    }
}

export function initializeFileExplorer() {
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
                displayMessage(`Dossier sélectionné: ${currentPath}`, 'info');
            }
        };
    }
}

export async function loadDirectoryContent(path) {
    try {
        const response = await fetch('/api/browse_files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path })
        });
        
        const data = await response.json();

        setCurrentPath(path);
        
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

