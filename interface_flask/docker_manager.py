import subprocess
import time
import os
import docker
import yaml
from typing import Dict, List, Optional
from pathlib import Path


class DockerComposeManager:
    def __init__(self, compose_file_path: str = "../docker-compose.yml"):
        """
        Initialise le gestionnaire Docker Compose.
        Args:
            compose_file_path: Chemin vers le fichier docker-compose.yml
        """
        self.compose_file_path = str(Path(compose_file_path).resolve())
        self.docker_client = docker.from_env()
        if not os.path.exists(compose_file_path):
            raise FileNotFoundError(f"Fichier docker-compose non trouvé: {compose_file_path}")

    def get_compose_services(self) -> List[str]:
        """Récupère la liste des services définis dans docker-compose.yml"""
        try:
            with open(self.compose_file_path, 'r') as file:
                compose_data = yaml.safe_load(file)
                services = list(compose_data.get('services', {}).keys())
                return services
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du docker-compose.yml: {e}")
            return []

    def run_compose_command(self, command: List[str]) -> tuple:
        """
        Exécute une commande docker-compose.

        Returns:
            tuple: (success: bool, output: str, error: str)
        """
        try:
            full_command = ['docker', 'compose', '-f', self.compose_file_path] + command

            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(self.compose_file_path))
            )

            return result.returncode == 0,  result.stdout or "", result.stderr or ""

        except Exception as e:
            return False, "", str(e)

    def start_all_services(self) -> List[str]:
        """Démarre tous les services définis dans docker-compose.yml"""
        messages = []
        messages.append("🚀 Démarrage de tous les services Docker Compose...")
        success, output, error = self.run_compose_command(['up', '-d'])
        if success:
            messages.append("✅ Tous les services ont été démarrés avec succès")
            if output and output.strip():
                messages.append(f"📝 Sortie: {output.strip()}")
        else:
            messages.append(f"❌ Erreur lors du démarrage des services: {error}")
        return messages

    def start_service(self, service_name: str) -> List[str]:
        """Démarre un service spécifique"""
        messages = []
        messages.append(f"🚀 Démarrage du service '{service_name}'...")

        services = self.get_compose_services()
        if service_name not in services:
            messages.append(f"❌ Service '{service_name}' non trouvé dans docker-compose.yml")
            messages.append(f"📋 Services disponibles: {', '.join(services)}")
            return messages
        success, output, error = self.run_compose_command(['up', '-d', service_name])
        if success:
            messages.append(f"✅ Service '{service_name}' démarré avec succès")
            if output and output.strip():
                messages.append(f"📝 Sortie: {output.strip()}")
        else:
            messages.append(f"❌ Erreur lors du démarrage du service '{service_name}': {error}")
        return messages

    def stop_service(self, service_name: str) -> List[str]:
        """Arrête un service spécifique"""
        messages = []
        messages.append(f"🛑 Arrêt du service '{service_name}'...")
        success, output, error = self.run_compose_command(['stop', service_name])
        if success:
            messages.append(f"✅ Service '{service_name}' arrêté avec succès")
        else:
            messages.append(f"❌ Erreur lors de l'arrêt du service '{service_name}': {error}")
        return messages

    def stop_all_services(self) -> List[str]:
        """Arrête tous les services"""
        messages = []
        messages.append("🛑 Arrêt de tous les services Docker Compose...")
        success, output, error = self.run_compose_command(['down'])
        if success:
            messages.append("✅ Tous les services ont été arrêtés avec succès")
        else:
            messages.append(f"❌ Erreur lors de l'arrêt des services: {error}")
        return messages

    def get_services_status(self) -> Dict[str, dict]:
        services_status = {}
        services = self.get_compose_services()
        containers = self.docker_client.containers.list(all=True)

        for service_name in services:
            container = None
            for cont in containers:
                if cont.labels.get("com.docker.compose.service") == service_name:
                    container = cont
                    break
            if container:
                services_status[service_name] = {
                    'running': container.status == 'running',
                    'status': container.status,
                    'container_name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown'
                }
            else:
                services_status[service_name] = {
                    'running': False,
                    'status': 'not_found',
                    'container_name': None,
                    'image': None
                }

        return services_status

    def restart_service(self, service_name: str) -> List[str]:
        """Redémarre un service spécifique"""
        messages = []
        messages.append(f"🔄 Redémarrage du service '{service_name}'...")
        success, output, error = self.run_compose_command(['restart', service_name])
        if success:
            messages.append(f"✨ Service '{service_name}' redémarré avec succès")
        else:
            messages.append(f"❌ Erreur lors du redémarrage du service '{service_name}': {error}")
        return messages

    def get_service_logs(self, service_name: str, lines: int = 50) -> List[str]:
        """Récupère les logs d'un service"""
        messages = []

        success, output, error = self.run_compose_command(['logs', '--tail', str(lines), service_name])

        if success:
            messages.append(f"📋 Logs du service '{service_name}' (dernières {lines} lignes):")
            if output.strip():
                for line in output.strip().split('\n'):
                    messages.append(f"  {line}")
            else:
                messages.append("  (Aucun log disponible)")
        else:
            messages.append(f"❌ Erreur lors de la récupération des logs: {error}")
        return messages

    def check_compose_file(self) -> List[str]:
        """Vérifie la validité du fichier docker-compose.yml"""
        messages = []
        success, output, error = self.run_compose_command(['config'])
        if success:
            messages.append("✅ Fichier docker-compose.yml valide")
        else:
            messages.append("❌ Erreur dans le fichier docker-compose.yml:")
            messages.append(f"  {error}")
        return messages


def create_docker_manager(compose_path: str = "docker-compose.yml"):
    """Factory function pour créer un gestionnaire Docker Compose"""
    try:
        return DockerComposeManager(compose_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None


def start_jupyter_service(compose_manager: DockerComposeManager, service_name: str = "jupyter") -> List[str]:
    """Démarre spécifiquement le service Jupyter"""
    messages = []
    status = compose_manager.get_services_status()

    if service_name in status:
        if status[service_name]['running']:
            messages.append(f"🟢 Service '{service_name}' déjà en cours d'exécution")
        else:
            messages.extend(compose_manager.start_service(service_name))
            time.sleep(5)
    else:
        messages.append(f"❌ Service '{service_name}' non trouvé dans docker-compose.yml")
    return messages
