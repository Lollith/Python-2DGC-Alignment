
import os
import ipywidgets as widgets
from IPython.display import display
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
from identification import sample_identification
import netCDF4 as nc

def get_mod_time(file_path):
    data = nc.Dataset(file_path, 'r',encoding="latin-1")
    scan_number = data.dimensions['scan_number'].size
    if scan_number == 328125:   
        mod_time = 1.25
        print("type de donnees: G0/plasma")
    elif scan_number == 540035:
        mod_time = 1.7
        print("type de donnnees: air expire")
    else:
        print("scan_number non reconnu")
    return mod_time
    

def analyse(path, output_path, method, mode, noise_factor,
                min_persistence, hit_prob_min, abs_threshold, rel_threshold, cluster, min_distance,
                min_sigma, max_sigma, sigma_ratio, num_sigma, formated_spectra, match_factor_min):
        """Run the analysis on the specified files."""
        if not path:
            print("Erreur : Aucun chemin sélectionné.")
            return
        

        files_list = os.listdir(path)
        files_list=[file for file in files_list if ('.cdf' in file)]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
        
        files_list = [file.strip() for file in files_list if file.strip()]
        print(f"Fichiers à analyser : {files_list}")
      
        for file in files_list:
            full_path = os.path.join(path, file)
            if not os.path.isfile(full_path):
                print(f"Erreur : Le fichier '{file}' est introuvable dans '{path}'")
                return
            if not os.access(full_path, os.R_OK):
                print(f"Erreur: Permission refusée pour accéder à '{file}' dans '{path}'")
                return
            
            mod_time = get_mod_time(full_path)
            
            print(f"Analyzing {file} with modulation time = {mod_time} secondes...\n")
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
                min_persistence
                )
            print("Analyse terminée:", result)
        print("Tous les fichiers ont été analysés avec succès.")


if __name__ == '__main__':

    noise_factor=1.5
    abs_threshold=0
    rel_threshold=0.01
    min_persistence = 0.0002
    method="persistent_homology"
    mode="tic"
    match_factor_min = 700
    hit_prob_min=700

    cluster=True
    min_distance= 1
    min_sigma= 1
    max_sigma= 30
    sigma_ratio= 1.6
    num_sigma= 1
    formated_spectra = 1 
    dir=["D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_01/", 
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_02/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_03/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_04/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_05/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_06/exported/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Elo/VOC-compare/cdf centroid/Tedlar/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Elo/VOC-compare/cdf centroid/ReCIVA/",
         "D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Elo/VOC-compare/cdf centroid/Blank/"]
    
    dir=["D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_06/exported/"]  
    for path in dir :
        outputpath= path + "output_python2dgc/"
        analyse(path, 
            outputpath, 
            method, mode, noise_factor, min_persistence, hit_prob_min, abs_threshold, rel_threshold, cluster, min_distance,
                min_sigma, max_sigma, sigma_ratio, num_sigma, formated_spectra, match_factor_min)