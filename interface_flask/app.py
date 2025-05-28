import os
from datetime import datetime
from waitress import serve
from flask import Flask, render_template, request, jsonify, send_file, Response
from data_converter import DataConverter
from datetime import datetime
import os
import time
import sys

app = Flask(__name__)
USERNAME = 'admin'
PASSWORD = 'MasSpec'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'converted_data'
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024  # 3GB max file size

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Instance globale du convertisseur
converter = DataConverter()


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
    """API pour lister les fichiers CDF dans un dossier."""
    data = request.get_json()
    path = data.get('path', '')

    if not path or not os.path.isdir(path):
        return jsonify({'success': False, 'message': 'Chemin invalide'})

    files = converter.get_files_from_folder(path)
    return jsonify({'success': True, 'files': files})


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
    import shutil
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


@app.route('/api/download/<path:filename>')
def download_file(filename):
    """API pour télécharger un fichier converti."""
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    # Configuration pour les gros fichiers
    
    # Augmenter la limite de récursion si nécessaire
    sys.setrecursionlimit(10000)
    
    # Configuration Flask pour gros fichiers
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    print("🚀 Serveur Flask démarré")
    print("📁 Dossier d'upload:", app.config['UPLOAD_FOLDER'])
    print("💾 Dossier de sortie:", app.config['OUTPUT_FOLDER'])
    print("⚠️  Limite de taille fichier: 3GB")
    print("🌐 Accédez à: http://localhost:5000")


    # Pour le développement, décommentez la ligne suivante :
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

    #TODO pour la Prod
    # serve(app, host='0.0.0.0', port=5000)