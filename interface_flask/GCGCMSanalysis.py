import os
import netCDF4 as nc
from identification import sample_identification


class GCGCMSAnalysis:
    """
    Flask version of the GC*GC_MS Analysis interface.
    Handles GCGCMS analysis through web interface.
    """
    
    def __init__(self):
        """Initialize the GCMS Analysis with default parameters."""
        # Environment setup
        self.docker_volume_path = os.getenv('DOCKER_VOLUME_PATH')
        self.host_volume_path = os.getenv('HOST_VOLUME_PATH')
        
        self.default_path_input = self.host_volume_path
        self.default_path_output = f'{self.host_volume_path}/output/'

        # Default values
        self.abs_threshold = "0"
        self.rel_threshold = "0.01"
        self.noise_factor = "1.5"
        self.min_persistence = "0.02"
        
        # Private parameters
        self._min_distance = 1
        self._sigma_ratio = 1.6
        self._num_sigma = 10
        self._min_sigma = 1
        self._max_sigma = 30
        self._overlap = 0.5
        self._match_factor_min = 650
        self._hit_prob_min = 15
        self._cluster = True
        self._min_samples = 4
        self._eps = 3

    # def get_files_from_folder(self, path):
    #     """Get all CDF files from a folder."""
    #     if os.path.isdir(path):
    #         return [f for f in os.listdir(path) if f.endswith(".cdf")]
    #     else:
    #         return []
    
    def get_mod_time(self, file_path):
        """Get modulation time based on scan_number from CDF file."""
        data = nc.Dataset(file_path, 'r')
        scan_number = data.dimensions['scan_number'].size
        data.close()
        
        if scan_number == 328125:   
            mod_time = 1.25
            print("type de donnees: G0/plasma")
        elif scan_number == 540035:
            mod_time = 1.7
            print("type de donnnees: air expire")
        else:
            print("scan_number non reconnu")
            mod_time = 1.0  # Default value
        return mod_time
    
    def analyse(self, path, files_list, output_path, user_output_path, method, mode, noise_factor,
                min_persistence, hit_prob_min, abs_threshold, rel_threshold, cluster, min_distance,
                min_sigma, max_sigma, sigma_ratio, num_sigma, formated_spectra, match_factor_min,
                overlap, eps, min_samples):
        """Run the analysis on the specified files."""

        if not path:
            return {"status": "error", "message": "Erreur : Aucun chemin sélectionné."}
        
        if files_list is None:
            files_list = self.get_files_from_folder(path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {user_output_path}")
        
        files_list = [file.strip() for file in files_list if file.strip()]
        print(f"Fichiers à analyser : {files_list}")
      
        results = []
        for file in files_list:
            full_path = os.path.join(path, file)
            if not os.path.isfile(full_path):
                return {"status": "error", "message": f"Erreur : Le fichier '{file}' est introuvable dans '{path}'"}
            if not os.access(full_path, os.R_OK):
                return {"status": "error", "message": f"Erreur: Permission refusée pour accéder à '{file}' dans '{path}'"}
            
            mod_time = self.get_mod_time(full_path)
            
            print(f"Analyzing {file} with modulation time = {mod_time} secondes...\n")
            
       
            try:
                result = sample_identification(
                    path,
                    file,
                    output_path,
                    mod_time,
                    method,
                    mode,
                    noise_factor,
                    hit_prob_min,
                    abs_threshold,
                    rel_threshold,
                    cluster,
                    min_distance,
                    min_sigma,
                    max_sigma,
                    sigma_ratio,
                    num_sigma,
                    formated_spectra,
                    match_factor_min,
                    min_persistence,
                    overlap,
                    eps,
                    min_samples,
                )
                result = f"Analysis completed for {file}"  # Placeholder
                results.append({"file": file, "result": result, "status": "success"})
                print("Analyse terminée:", result)
            except Exception as e:
                results.append({"file": file, "result": str(e), "status": "error"})
        
        return {"status": "success", "message": "Tous les fichiers ont été analysés avec succès.", "results": results}
