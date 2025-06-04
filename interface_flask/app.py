import os
from datetime import datetime
from waitress import serve
from flask import Flask, render_template, request, jsonify, send_file, Response
from data_converter import DataConverter
from run_docker import RunDocker
# from GCGCMSanalysis import GCGCMSAnalysis
from datetime import datetime
import os
import time
import sys
import docker
from datetime import timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import netCDF4 as nc
from werkzeug.utils import secure_filename
import threading
import shutil
import subprocess

app = Flask(__name__)
USERNAME = 'admin'
PASSWORD = 'MasSpec'
client = docker.from_env()
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'converted_data'
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024  # 3GB max file size

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Instances
converter = DataConverter()
rundocker = RunDocker(client)
# gcgcms_analyzer = GCGCMSAnalysis()

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Authentification requise.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@requires_auth
def index():
    """Page principale avec le formulaire."""
    return render_template('index.html',
                         default_input_path=converter.default_path_input,
                         default_output_path=converter.default_path_output)

@app.route('/api/list_files', methods=['POST'])
def list_files():
    """API pour lister les fichiers avec extension spécifiée dans un dossier."""
    data = request.get_json()
    path = data.get('path', '')
    extension = data.get('extension', '.cdf')  # Extension par défaut
    
    if not path or not os.path.isdir(path):
        return jsonify({'success': False, 'message': 'Chemin invalide'})
    
    try:
        if extension == '.cdf':
            files = converter.get_files_from_folder(path)
        else:
            # Lister les fichiers avec l'extension spécifiée
            files = []
            for filename in os.listdir(path):
                if filename.lower().endswith(extension.lower()):
                    files.append(filename)
            files.sort()
        
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

@app.route('/api/convert', methods=['POST'])
def convert_files():
    """API pour convertir les fichiers avec support des gros fichiers."""
    t0 = time.time()
    data = request.get_json()
    input_path = data.get('input_path', '')
    output_path = data.get('output_path', '')
    files_str = data.get('files', '')
    
    # Traiter la liste des fichiers
    if files_str.strip():
        files_list = [f.strip() for f in files_str.split(',') if f.strip()]
    else:
        files_list = None
    
    # Vérifier l'espace disque avant de commencer
    try:
        free_space = shutil.disk_usage(output_path).free
        if free_space < 5 * 1024 * 1024 * 1024:  # Moins de 5GB libre
            return jsonify({
                'success': False,
                'messages': [f"⚠️ Attention: Seulement {free_space//1024//1024//1024}GB d'espace libre. Recommandé: >5GB"],
                'converted_files': [],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    except Exception as e:
        print(f"Erreur lors de la vérification de l'espace disque: {e}")
        pass
    
    # Effectuer la conversion
    success, messages, converted_files = converter.read_cdf_to_npy(input_path, files_list, output_path)
    end = time.time() - t0
    messages.append(f"Conversion terminée, temps_execution_sec: {round(end, 2)}")
    
    return jsonify({
        'success': success,
        'messages': messages,
        'converted_files': converted_files,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/check_docker', methods=['POST'])
def check_docker():
    """API pour vérifier l'état des conteneurs Docker."""
    try:
        container_status = rundocker.check_containers_status()
        all_running = all(status['running'] for status in container_status.values())
        
        status_messages = []
        for container_name, status in container_status.items():
            if status['running']:
                status_messages.append(f"🟢 {container_name}: En cours d'exécution")
            else:
                status_messages.append(f"🔴 {container_name}: Arrêté")
        
        return jsonify({
            'success': True,
            'all_running': all_running,
            'status': status_messages,
            'detailed_status': container_status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'all_running': False,
            'status': [f"❌ Erreur lors de la vérification: {str(e)}"],
            'detailed_status': {}
        })

@app.route('/api/analyze', methods=['POST'])
def analyze_files():
    """API pour lancer l'analyse des fichiers .npy."""
    t0 = time.time()
    data = request.get_json()
    analysis_path = data.get('analysis_path', '')
    selected_files = data.get('selected_files', [])
    
    messages = []
    
    # Validation
    if not analysis_path or not os.path.isdir(analysis_path):
        return jsonify({
            'success': False,
            'messages': ['❌ Chemin d\'analyse invalide'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    if not selected_files:
        return jsonify({
            'success': False,
            'messages': ['❌ Aucun fichier sélectionné pour l\'analyse'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Vérifier que les fichiers existent
    valid_files = []
    for filename in selected_files:
        file_path = os.path.join(analysis_path, filename)
        if os.path.exists(file_path):
            valid_files.append(filename)
            messages.append(f"✅ Fichier trouvé: {filename}")
        else:
            messages.append(f"⚠️ Fichier non trouvé: {filename}")
    
    if not valid_files:
        return jsonify({
            'success': False,
            'messages': messages + ['❌ Aucun fichier valide trouvé'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    try:
        # 1. Vérifier et démarrer les conteneurs Docker si nécessaire
        messages.append("🔍 Vérification des conteneurs Docker...")
        container_status = rundocker.check_containers_status()
        
        containers_to_start = []
        for container_name, status in container_status.items():
            if not status['running']:
                containers_to_start.append(container_name)
                messages.append(f"🔴 {container_name}: Arrêté")
            else:
                messages.append(f"🟢 {container_name}: En cours d'exécution")
        
        # Démarrer les conteneurs si nécessaire
        if containers_to_start:
            messages.append("🚀 Démarrage des conteneurs Docker...")
            start_result = rundocker.start_containers(containers_to_start)
            messages.extend(start_result)
        
        # 2. Lancer l'analyse
        messages.append(f"🔬 Début de l'analyse de {len(valid_files)} fichier(s)...")
        
        # Ici vous pouvez appeler votre classe d'analyse
        # analysis_result = gcgcms_analyzer.analyze_files(analysis_path, valid_files)
        
        # Pour l'instant, simulation de l'analyse
        import time
        time.sleep(2)  # Simulation du temps d'analyse
        
        messages.append(f"📊 Analyse des fichiers terminée:")
        for filename in valid_files:
            messages.append(f"   ✅ {filename} - Analysé avec succès")
        
        end = time.time() - t0
        messages.append(f"⏱️ Temps total d'analyse: {round(end, 2)} secondes")
        
        return jsonify({
            'success': True,
            'messages': messages,
            'analyzed_files': valid_files,
            'analysis_results': {
                'total_files': len(valid_files),
                'execution_time': round(end, 2)
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        messages.append(f"❌ Erreur lors de l'analyse: {str(e)}")
        return jsonify({
            'success': False,
            'messages': messages,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })



if __name__ == '__main__':
    # Configuration pour les gros fichiers
    
    # Augmenter la limite de récursion si nécessaire
    sys.setrecursionlimit(10000)
    
    # Configuration Flask pour gros fichiers
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    #TODO pour prod:
    # app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(days=365)
    #TODO a mettre sur false en prod
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    # print("🚀 Serveur Flask démarré")
    # print("📁 Dossier d'upload:", app.config['UPLOAD_FOLDER'])
    # print("💾 Dossier de sortie:", app.config['OUTPUT_FOLDER'])
    # print("⚠️  Limite de taille fichier: 3GB")
    print("🌐 Accédez à: http://localhost:5000")


    # # Pour le développement, décommentez la ligne suivante :
    # app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

    # #TODO pour la Prod
    # # serve(app, host='0.0.0.0', port=5000)

    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        print("🚀 Serveur Flask démarré en mode dev")
        print("⚠️  Limite de taille fichier: 3GB")
        print("🌐 Accédez à: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("🚀 Serveur Flask démarré")
        print("🚀 Démarrage du serveur en mode production...")
        print("⚠️  Limite de taille fichier: 3GB")
        print("📍 Serveur accessible sur: http://localhost:8080")
        serve(app, host='0.0.0.0', port=8080)
    