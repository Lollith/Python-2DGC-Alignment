import os
from datetime import datetime
from waitress import serve
from flask import Flask, render_template, request, jsonify, send_file, Response
from data_converter import DataConverter
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
import requests
import webbrowser
from functools import wraps
import docker_manager
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import logging
from flask import Flask, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash
import nist_engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# nist = nist_engine.NistEngine()#DEBUG

load_dotenv()
auth = HTTPBasicAuth()

app = Flask(__name__)

# USERNAME = os.getenv("FLASK_USERNAME")
# PASSWORD = os.getenv("FLASK_PASSWORD")
hashed_password = os.getenv('FLASK_HASHED_PASSWORD')
username_env = os.getenv('USERNAME')
ip_server = os.getenv("IP_SERVER")
docker_volume_path = os.getenv("DOCKER_VOLUME_PATH")

client = docker.from_env()

nist_executor = ThreadPoolExecutor(max_workers=8)

app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024  # 3GB max file size


# Instances
converter = DataConverter()
compose_manager = docker_manager.create_docker_manager("../docker-compose.dev.yml") #DEBUG
# nist_wrapper = nist_search.NISTSearchWrapper()


#def check_auth(username, password):
 #    return username == USERNAME and password == PASSWORD

@auth.verify_password
def verify_password(username, password):
    return username == username_env and check_password_hash(hashed_password, password)

# @app.route('/nist/health')
# @auth.login_required
# def nist_health():
#     return jsonify({"status": "available"})

# def authenticate():
#     return Response(
#         'Authentification requise.', 401,
#         {'WWW-Authenticate': 'Basic realm="Login Required"'})


# def requires_auth(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         auth = request.authorization
#         if not auth or not check_auth(auth.username, auth.password):
#             return authenticate()
#         return f(*args, **kwargs)
#     return decorated

@auth.login_required
@app.route('/')
def index():
    """Page principale avec le formulaire."""
    return render_template('index.html',
                           default_input_path=converter.default_path_input,
                           default_output_path=converter.default_path_output,
                           docker_volume_path=docker_volume_path
                           )


# @app.route('/api/list_files', methods=['POST'])
# def list_files():
#     """API pour lister les fichiers avec extension spécifiée dans un dossier."""
#     data = request.get_json()
#     path = data.get('path', '')
#     extension = data.get('extension', '.cdf')  # Extension par défaut

#     if not path or not os.path.isdir(path):
#         return jsonify({'success': False, 'message': 'Chemin invalide'})

#     try:
#         if extension == '.cdf':
#             files = converter.get_files_from_folder(path)
#         else:
#             files = []
#             for filename in os.listdir(path):
#                 if filename.lower().endswith(extension.lower()):
#                     files.append(filename)
#             files.sort()

#         return jsonify({'success': True, 'files': files})
#     except Exception as e:
#         return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})


@app.route('/api/list_files', methods=['POST'])
def list_files():
    data = request.get_json()
    path = data.get('path', '')
    extension = data.get('extension', '.cdf')

    if not path or not os.path.isdir(path):
        return jsonify({'success': False, 'message': 'Chemin invalide'})

    try:
        folders, files = [], []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                folders.append({'name': entry, 'path': full_path})
            elif entry.lower().endswith(extension.lower()):
                files.append({'name': entry, 'path': full_path})

        folders.sort(key=lambda x: x['name'])
        files.sort(key=lambda x: x['name'])

        return jsonify({'success': True, 'folders': folders, 'files': files})
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
    success, messages, converted_files = (
        converter.convert_cdf_to_hdf5_threaded(
            input_path, files_list, output_path
            ))
    # end = time.time() - t0
    # messages.append(f"Conversion terminée, temps_execution_sec: {round(end, 2)}")

    return jsonify({
        'success': success,
        'messages': messages,
        'converted_files': converted_files,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/check_containers', methods=['POST'])
def check_containers():
    if compose_manager is None:
        return jsonify({
            'success': False,
            'all_running': False,
            'status': ["❌ Gestionnaire Docker Compose non initialisé"],
            'detailed_status': {},
        })
    try:
        services_status = compose_manager.get_services_status()
        all_running = all(status['running'] for status in services_status.values())

        status_messages = []
        for container_name, status in services_status.items():
            print("status", status)
            if status['running']:
                status_messages.append(f"🟢 {container_name}: En cours d'exécution")
            else:
                status_messages.append(f"🔴 {container_name}: Arrêté ({status['status']})")

        return jsonify({
            'success': True,
            'all_running': all_running,
            'status': status_messages,
            'detailed_status': services_status,
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'all_running': False,
            'status': [f"❌ Erreur: {str(e)}"],
            'detailed_status': {},
        })

# @app.route('/api/start_containers', methods=['POST'])
# def start_containers():
#     if compose_manager is None:
#         return jsonify({
#             'success': False,
#             'all_running': False,
#             'status': ["❌ Gestionnaire Docker Compose non initialisé"],
#             'detailed_status': {}
#         })
#     try:
#         services_status = compose_manager.get_services_status()
#         all_running = all(status['running'] for status in services_status.values())
#         status_messages = []
#         for container_name, status in services_status.items():
#             if status['running']:
#                 status_messages.append(f"🟢 {container_name}: En cours d'exécution")
#             else:
#                 status_messages.append(f"🔴 {container_name}: Arrêté ({status['status']})")
#                 start_messages = compose_manager.start_service(container_name)
#                 status_messages.extend(start_messages)

#         return jsonify({
#             'success': True,
#             'all_running': all_running,
#             'status': status_messages,
#             'detailed_status': services_status
#         })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'all_running': False,
#             'status': [f"❌ Erreur: {str(e)}"],
#             'detailed_status': {}
#         })

@app.route('/api/start_containers', methods=['POST'])
def start_containers():
    if compose_manager is None:
        return jsonify({
            'success': False,
            'status': ["❌ Gestionnaire Docker Compose non initialisé"]
        })

    def launch():
        try:
            for container_name in compose_manager.get_compose_services():
                compose_manager.start_service(container_name)
        except Exception as e:
            print("Erreur lors du lancement des conteneurs:", e)

    # Lancement en arrière-plan (ne bloque pas la réponse HTTP)
    threading.Thread(target=launch, daemon=True).start()

    return jsonify({
        "success": True,
        "status": ["🚀 Lancement des conteneurs demandé, vérifie l’état dans quelques secondes."]
    })


# @app.route('/api/docker-compose/stop', methods=['POST'])
# def stop_docker_services():
#     """API pour arrêter des services Docker Compose"""
#     if not compose_manager:
#         return jsonify({
#             'success': False,
#             'message': 'Gestionnaire Docker Compose non disponible'
#         })
    
#     data = request.get_json()
#     service_name = data.get('service_name', None)
    
#     if service_name:
#         # Arrêter un service spécifique
#         messages = compose_manager.stop_service(service_name)
#     else:
#         # Arrêter tous les services
#         messages = compose_manager.stop_all_services()
    
#     return jsonify({
#         'success': True,
#         'messages': messages,
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     })


# @app.route('/api/docker-compose/logs/<service_name>', methods=['GET'])
# def get_service_logs(service_name):
#     """API pour récupérer les logs d'un service"""
#     if not compose_manager:
#         return jsonify({
#             'success': False,
#             'message': 'Gestionnaire Docker Compose non disponible'
#         })
    
#     lines = request.args.get('lines', 50, type=int)
#     messages = compose_manager.get_service_logs(service_name, lines)
    
#     return jsonify({
#         'success': True,
#         'messages': messages,
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     })


@app.route('/api/analyze', methods=['POST'])
def analyze_files():
    """API pour lancer l'analyse des fichiers .h5."""
    data = request.get_json()
    analysis_path = data.get('analysis_path', '')
    selected_files = data.get('selected_files', [])

    messages = []

    # if not analysis_path or not os.path.isdir(analysis_path):
    #     return jsonify({
    #         'success': False,
    #         'messages': ['❌ Chemin d\'analyse invalide'],
    #         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     })

    # if not selected_files:
    #     return jsonify({
    #         'success': False,
    #         'messages': ['❌ Aucun fichier sélectionné pour l\'analyse'],
    #         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     })

    valid_files = []
    for filename in selected_files:
        file_path = os.path.join(analysis_path, filename)
        if os.path.exists(file_path):
            valid_files.append(filename)
            messages.append(f"✅ Fichier trouvé: {filename}")
        else:
            messages.append(f"⚠️ Fichier non trouvé: {filename}")

    # if not valid_files:
    #     return jsonify({
    #         'success': False,
    #         'messages': messages + ['❌ Aucun fichier valide trouvé'],
    #         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     })

    try:
        # 1. Vérifier et démarrer les conteneurs Docker si nécessaire
        messages.append("🔍 Vérification des conteneurs Docker...")
        # container_status = rundocker.check_containers_status()
        services_status = compose_manager.get_services_status()

        services_to_start = []
        for service_name, status in services_status.items():
            if status['running']:
                messages.append(f"🟢 {service_name}: En cours d'exécution")
            else:
                messages.append(f"🔴 {service_name}: Arrêté ({status['status']})")
                services_to_start.append(service_name)

        # Démarrer les conteneurs si nécessaire
        if services_to_start:
            messages.append("🚀 Démarrage des conteneurs Docker...")

            for service in services_to_start:
                start_messages = compose_manager.start_service(service)
                messages.extend(start_messages)

            # Attendre que les services soient complètement démarrés
            messages.append("⏳ Attente du démarrage complet des conteneurs...")
            time.sleep(5)
        
        # 2. Vérifier que Jupyter Lab est accessible et l'ouvrir
        jupyter_url = f"http://{ip_server}:8888/lab/tree/run_interfaces.ipynb"
        messages.append("🔍 Vérification de la disponibilité de Jupyter Lab...")
        # self.wait_and_open_jupyter() #TODO verifier ici

        def wait_and_open_jupyter():
            """Fonction pour attendre que Jupyter soit prêt et l'ouvrir"""
            max_attempts = 30  
            attempt = 0

            while attempt < max_attempts:
                try:
                    response = requests.get(jupyter_url, timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Jupyter Lab est accessible, ouverture du navigateur...")
                        webbrowser.open(jupyter_url)
                        break
                except requests.exceptions.RequestException:
                    pass

                attempt += 1
                time.sleep(1)

            if attempt >= max_attempts:
                print("❌ Impossible d'accéder à Jupyter Lab après 30 secondes")

        # Lancer la vérification et l'ouverture de Jupyter en arrière-plan
        jupyter_thread = threading.Thread(target=wait_and_open_jupyter)
        jupyter_thread.daemon = True
        jupyter_thread.start()

        messages.append(f"🌐 Ouverture de Jupyter Lab sur {jupyter_url}...")

        return jsonify({
            'success': True,
            'messages': messages,
            'analyzed_files': valid_files,
            'jupyter_url': jupyter_url,
            'analysis_results': {
                'total_files': len(valid_files),
            },

        })

    except Exception as e:
        messages.append(f"❌ Erreur lors de l'analyse: {str(e)}")
        return jsonify({
            'success': False,
            'messages': messages,
        })


def check_jupyter_health(url="http://localhost:8888", timeout=2):
    """Fonction utilitaire pour vérifier si Jupyter Lab est accessible"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@app.route('/api/jupyter-status', methods=['GET'])
def jupyter_status():
    """API pour vérifier le statut de Jupyter Lab"""
    jupyter_url = "http://localhost:8888"
    is_running = check_jupyter_health(jupyter_url)

    return jsonify({
        'jupyter_running': is_running,
        'jupyter_url': jupyter_url,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/api/open-jupyter', methods=['POST'])
def open_jupyter():
    """API pour ouvrir Jupyter Lab dans le navigateur"""
    jupyter_url = "http://localhost:8888"
    
    if check_jupyter_health(jupyter_url):
        webbrowser.open(jupyter_url)
        return jsonify({
            'success': True,
            'message': 'Jupyter Lab ouvert dans le navigateur',
            'jupyter_url': jupyter_url,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Jupyter Lab n\'est pas accessible',
            'jupyter_url': jupyter_url,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


######## NIST Search Endpoints ########

@app.route('/nist/health', methods=['GET'])
def nist_health():
    """Vérification NIST disponible"""
    return jsonify({
        'nist_status': 'available',
        'timestamp': time.time(),
        'active_threads': len(nist_executor._threads) if hasattr(nist_executor, '_threads') else 0
    })

# @app.route('/nist/search', methods=['POST'])
# def nist_single_search():
#     """Recherche NIST d'un spectre unique"""
#     try:
#         spectrum_data = request.json
        
#         if not spectrum_data:
#             return jsonify({'error': 'Données de spectre manquantes'}), 400
        
#         logger.info("Recherche NIST single spectre")
#         result = nist_wrapper.search_spectrum(spectrum_data)
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"Erreur NIST single search: {e}")
#         return jsonify({'error': str(e)}), 500
# from pyms.Spectrum import MassSpectrum

# @app.route('/nist/batch_search', methods=['POST'])
# def nist_batch_search():
#     """Recherche NIST en lot (optimisée)"""
#     try:
#         data = request.json
#         spectra = data.get('spectra', [])
        
#         if not spectra:
#             return jsonify({'error': 'Liste de spectres vide'}), 400
        
#         logger.info(f"Recherche NIST batch: {len(spectra)} spectres")
#         start_time = time.time()

#         def dict_to_mass_spectrum(spectrum_dict):
#             return MassSpectrum(
#                 mass_list=[float(m) for m in spectrum_dict["mass"]],
#                 intensity_list=[float(i) for i in spectrum_dict["intensity"]]
#             )

#         spectra_ms = [dict_to_mass_spectrum(s) for s in spectra]
        
#         # Traitement parallèle avec votre pool existant
#         future_to_index = {
#             nist_executor.submit(nist_wrapper.nist_batch_search, [spectrum]): i
#             for i, spectrum in enumerate(spectra_ms)
#         }
        
#         results = [None] * len(spectra)
#         completed = 0
        
#         for future in future_to_index:
#             index = future_to_index[future]
#             try:
#                 result = future.result()
#                 results[index] = result
#                 completed += 1
                
#                 if completed % 10 == 0:
#                     logger.info(f"NIST progression: {completed}/{len(spectra)}")
                    
#             except Exception as e:
#                 logger.error(f"Erreur spectre {index}: {e}")
#                 results[index] = {'error': str(e), 'hits': []}
        
#         total_time = time.time() - start_time
#         logger.info(f"NIST batch terminé: {len(spectra)} spectres en {total_time:.2f}s")
        
#         return jsonify({
#             'results': results,
#             'total_time': total_time,
#             'spectra_count': len(spectra),
#             'performance': f"{len(spectra)/total_time:.1f} spectres/sec"
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur NIST batch: {e}")
#         return jsonify({'error': str(e)}), 500


@app.route('/nist/search', methods=['POST'])
def nist_search():
    """
    Endpoint Flask pour un spectre unique.
    """
    print("Requête reçue :", request.json)
    try:
        data = request.json
        if not data or "mass" not in data or "intensity" not in data:
            return jsonify({"error": "Spectre invalide"}), 400

        logger.info("Recherche NIST pour un spectre")

        result = nist.full_search_with_ref_data(data)
        return jsonify({"hits": result})

    except Exception as e:
        logger.error(f"Erreur NIST search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'route': str(rule)
        })
    return jsonify(routes)


if __name__ == '__main__':

    # Augmenter la limite de récursion si nécessaire
    sys.setrecursionlimit(10000)

    # Configuration Flask pour gros fichiers
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    if len(sys.argv) > 1 and sys.argv[1] == 'dev':
        print("🚀 Serveur Flask démarré en mode dev")
        print("⚠️  Limite de taille fichier: 3GB")
        print("🌐 Accédez à: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("🚀 Serveur Flask démarré")
        print("🚀 Démarrage du serveur en mode production...")
        print("⚠️  Limite de taille fichier: 3GB")
        print(f"📍 Serveur accessible sur: http://{ip_server}:8080")
        logging.getLogger('waitress').setLevel(logging.WARNING)
        serve(app, host='0.0.0.0', port=8080)