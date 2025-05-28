import subprocess


class RunDocker():
    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.container_app = "2dgc_id_app"
        self.container_nist = "nist_engine"
        self.image_name_app = "python-2dgc-alignment-2dgc-id"
        self.image_name_nist = "python-2dgc-alignment-nist"

    def is_container_running(self, container_name):
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())

    def run_containers(self):
        """
        Run a Docker container with the specified image and parameters.
        
        :param image_name: Name of the Docker image to run.
        :param command: Command to run in the container.
        
        :return: The created container object.
        """
        messages = []
        if self.is_container_running(self.container_nist):
            messages.append(f"🟢 Container '{self.container_nist}' is already running.")
            return messages
        else:
            messages.append(f"🔵 Starting container '{self.container_nist}'...")
            return messages
        # try:
        #     # Récupère les conteneurs Docker actifs
        #     containers = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"]).decode()

        #     if service:
        #         if service not in containers:
        #             subprocess.run(["../docker-compose", "up", "-d", service], check=True)
        #             print(f"✅ Service Docker '{service}' lancé.")
        #         else:
        #             print(f"🟢 Service Docker '{service}' déjà en cours d'exécution.")
        #     else:
        #         subprocess.run(["../docker-compose", "up", "-d"], check=True)
        #         print("✅ Tous les services Docker sont lancés.")
        #     return True
        # except subprocess.CalledProcessError as e:
        #     print(f"❌ Erreur Docker : {e}")
        #     return False
        
    # def stop_container(self, container_id):
    #     """
    #     Stop a running Docker container.
    #     :param container_id: ID of the container to stop.
    #     """ 
    #     container = self.docker_client.containers.get(container_id)
    #     container.stop()
    #     container.remove()
    #     print(f"Container {container_id} stopped and removed.")
    #     return container
    # def list_containers(self):
    #     """
    #     List all running Docker containers.
    #     :return: List of running containers.
    #     """
    #     return self.docker_client.containers.list()
    # def get_container_logs(self, container_id):
    #     """
    #     Get logs from a specific Docker container.
    #     :param container_id: ID of the container to get logs from.
    #     :return: Logs from the container.
    #     """
    #     container = self.docker_client.containers.get(container_id)
    #     return container.logs().decode('utf-8')
    # def get_container_status(self, container_id):
    #     """
    #     Get the status of a specific Docker container.
    #     :param container_id: ID of the container to check status.
    #     :return: Status of the container.
    #     """
    #     container = self.docker_client.containers.get(container_id)
    #     return container.status
    # def get_container_info(self, container_id):
    #     """
    #     Get detailed information about a specific Docker container.
    #     :param container_id: ID of the container to get information about.
    #     :return: Information about the container.
    #     """         
    #     container = self.docker_client.containers.get(container_id)
    #     return {
    #         'id': container.id,
    #         'name': container.name,
    #         'status': container.status,
    #         'image': container.image.tags,
    #         'created': container.attrs['Created'],
    #         'ports': container.attrs['NetworkSettings']['Ports']
    #     }
    # def remove_container(self, container_id):
    #     """
    #     Remove a Docker container.
    #     :param container_id: ID of the container to remove.
    #     """     
    #     container = self.docker_client.containers.get(container_id)
    #     container.remove(force=True)
    #     print(f"Container {container_id} removed.")
    #     return container