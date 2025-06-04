import subprocess
import docker
import time

class RunDocker():
    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.container_app = "2dgc_id_app"
        self.container_nist = "nist_engine"
        self.image_name_app = "python-2dgc-alignment-2dgc-id"
        self.image_name_nist = "python-2dgc-alignment-nist"
        
        # Mapping des conteneurs et leurs images
        self.containers = {
            self.container_app: self.image_name_app,
            self.container_nist: self.image_name_nist
        }

    def is_container_running(self, container_name):
        """Vérifier si un conteneur spécifique est en cours d'exécution."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container_name}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Fallback avec l'API Docker si la commande échoue
            try:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if container_name in container.name and container.status == 'running':
                        return True
                return False
            except Exception:
                return False

    def check_containers_status(self):
        """Vérifier l'état de tous les conteneurs."""
        status = {}
        for container_name in self.containers.keys():
            try:
                is_running = self.is_container_running(container_name)
                status[container_name] = {
                    'running': is_running,
                    'status': 'running' if is_running else 'stopped'
                }
            except Exception as e:
                status[container_name] = {
                    'running': False,
                    'status': f'error: {str(e)}'
                }
        return status

    def start_container(self, container_name):
        """Démarrer un conteneur spécifique."""
        try:
            # Vérifier si le conteneur existe déjà
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == 'running':
                    return f"🟢 Container '{container_name}' is already running."
                else:
                    container.start()
                    # Attendre un peu pour que le conteneur démarre
                    time.sleep(2)
                    return f"🟢 Container '{container_name}' started successfully."
            except docker.errors.NotFound:
                # Le conteneur n'existe pas, le créer
                image_name = self.containers.get(container_name)
                if not image_name:
                    return f"❌ Unknown container '{container_name}'"
                
                # Créer et démarrer le conteneur
                container = self.docker_client.containers.run(
                    image_name,
                    name=container_name,
                    detach=True,
                    # Ajoutez ici d'autres paramètres selon vos besoins
                    # ports={'5000/tcp': 5000},  # exemple
                    # volumes={'/host/path': {'bind': '/container/path', 'mode': 'rw'}},
                )
                time.sleep(2)
                return f"🟢 Container '{container_name}' created and started successfully."
                
        except Exception as e:
            return f"❌ Error starting container '{container_name}': {str(e)}"

    def start_containers(self, container_names=None):
        """Démarrer une liste de conteneurs ou tous les conteneurs."""
        if container_names is None:
            container_names = list(self.containers.keys())
        
        messages = []
        for container_name in container_names:
            message = self.start_container(container_name)
            messages.append(message)
        
        return messages

    def stop_container(self, container_name):
        """Arrêter un conteneur spécifique."""
        try:
            container = self.docker_client.containers.get(container_name)
            if container.status == 'running':
                container.stop()
                return f"🛑 Container '{container_name}' stopped successfully."
            else:
                return f"ℹ️ Container '{container_name}' is not running."
        except docker.errors.NotFound:
            return f"⚠️ Container '{container_name}' not found."
        except Exception as e:
            return f"❌ Error stopping container '{container_name}': {str(e)}"

    def restart_container(self, container_name):
        """Redémarrer un conteneur spécifique."""
        try:
            container = self.docker_client.containers.get(container_name)
            container.restart()
            time.sleep(2)
            return f"🔄 Container '{container_name}' restarted successfully."
        except docker.errors.NotFound:
            return f"⚠️ Container '{container_name}' not found."
        except Exception as e:
            return f"❌ Error restarting container '{container_name}': {str(e)}"

    def get_container_logs(self, container_name, lines=50):
        """Récupérer les logs d'un conteneur."""
        try:
            container = self.docker_client.containers.get(container_name)
            logs = container.logs(tail=lines).decode('utf-8')
            return {
                'success': True,
                'logs': logs,
                'container': container_name
            }
        except docker.errors.NotFound:
            return {
                'success': False,
                'error': f"Container '{container_name}' not found.",
                'container': container_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error getting logs for '{container_name}': {str(e)}",
                'container': container_name
            }

    def run_containers(self):
        """
        Méthode héritée - démarre tous les conteneurs et retourne les messages.
        Utilisée pour la rétrocompatibilité avec votre code existant.
        """
        messages = []
        
        for container_name in self.containers.keys():
            if self.is_container_running(container_name):
                messages.append(f"🟢 Container '{container_name}' is already running.")
            else:
                messages.append(f"🔵 Starting container '{container_name}'...")
                start_message = self.start_container(container_name)
                messages.append(start_message)
        
        return messages

    def get_containers_info(self):
        """Récupérer des informations détaillées sur tous les conteneurs."""
        info = {}
        for container_name in self.containers.keys():
            try:
                container = self.docker_client.containers.get(container_name)
                info[container_name] = {
                    'name': container.name,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'created': container.attrs['Created'],
                    'ports': container.ports,
                    'running': container.status == 'running'
                }
            except docker.errors.NotFound:
                info[container_name] = {
                    'name': container_name,
                    'status': 'not_found',
                    'image': self.containers[container_name],
                    'created': None,
                    'ports': {},
                    'running': False
                }
            except Exception as e:
                info[container_name] = {
                    'name': container_name,
                    'status': f'error: {str(e)}',
                    'image': self.containers[container_name],
                    'created': None,
                    'ports': {},
                    'running': False
                }
        
        return info