import os
from datetime import datetime
from waitress import serve
from flask import Flask, render_template, request, jsonify, send_file, Response
from data_converter import DataConverter
from run_docker import RunDocker
from GCGCMSanalysis import GCGCMSAnalysis
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
gcgcms_analyzer = GCGCMSAnalysis()


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

@app.route('/next')
def next_page():
    return render_template('next.html',default_path=gcgcms_analyzer.default_path_input,
                         default_output_path=gcgcms_analyzer.default_path_output,
                        #  available_files=available_files,
                         default_noise_factor=gcgcms_analyzer.noise_factor,
                         default_min_persistence=gcgcms_analyzer.min_persistence,
                         default_abs_threshold=gcgcms_analyzer.abs_threshold,
                         default_rel_threshold=gcgcms_analyzer.rel_threshold)

@app.route('/next/analyze', methods=['POST'])
def analyze():
    """Process the analysis request."""
    messages = []
    runsuccess = rundocker.run_containers()
    messages.extend(runsuccess)

    # Get form data
    user_input_path = request.form.get(
        'path', gcgcms_analyzer.default_path_input)
    path_for_docker = user_input_path.replace(
        gcgcms_analyzer.host_volume_path,
        gcgcms_analyzer.docker_volume_path, 1)

    user_output_path = request.form.get('output_path', gcgcms_analyzer.default_path_output)
    output_path_for_docker = user_output_path.replace(gcgcms_analyzer.host_volume_path, 
                                                        gcgcms_analyzer.docker_volume_path, 1)
    
    files_input = request.form.get('files', '')
    files_list = [f.strip() for f in files_input.split(",") if f.strip()] if files_input else None
    
    method = request.form.get('method', 'persistent_homology')
    mode = request.form.get('mode', 'tic')
    noise_factor = float(request.form.get('noise_factor', gcgcms_analyzer.noise_factor))
    min_persistence = float(request.form.get('min_persistence', gcgcms_analyzer.min_persistence))
    abs_threshold = float(request.form.get('abs_threshold', gcgcms_analyzer.abs_threshold))
    rel_threshold = float(request.form.get('rel_threshold', gcgcms_analyzer.rel_threshold))
    formated_spectra = request.form.get('formated_spectra') == 'on'
    
    # Run analysis
    result = gcgcms_analyzer.analyse(
        path_for_docker, files_list, output_path_for_docker, user_output_path,
        method, mode, noise_factor, min_persistence, gcgcms_analyzer._hit_prob_min,
        abs_threshold, rel_threshold, gcgcms_analyzer._cluster,
        gcgcms_analyzer._min_distance, gcgcms_analyzer._min_sigma, gcgcms_analyzer._max_sigma, 
        gcgcms_analyzer._sigma_ratio, gcgcms_analyzer._num_sigma, formated_spectra,
        gcgcms_analyzer._match_factor_min, gcgcms_analyzer._overlap, 
        gcgcms_analyzer._eps, gcgcms_analyzer._min_samples
    )
    
    return render_template('results.html', messages=result)
    # return jsonify({'messages': messages,})   


@app.route('/next/analyze_async', methods=['POST'])
def analyze_async():
    """Start analysis in background and return immediately."""
    try:
        # This would be useful for long-running analyses
        # You could implement a job queue system here
        return jsonify({"status": "started", "message": "Analysis started in background"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


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
    
    print("🚀 Serveur Flask démarré")
    print("📁 Dossier d'upload:", app.config['UPLOAD_FOLDER'])
    print("💾 Dossier de sortie:", app.config['OUTPUT_FOLDER'])
    print("⚠️  Limite de taille fichier: 3GB")
    print("🌐 Accédez à: http://localhost:5000")


    # Pour le développement, décommentez la ligne suivante :
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

    #TODO pour la Prod
    # serve(app, host='0.0.0.0', port=5000)