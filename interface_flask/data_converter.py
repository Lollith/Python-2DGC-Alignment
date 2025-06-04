# import netCDF4 as nc
# import os
# import numpy as np


# def read_cdf_to_npy(self, path, files_list, output_path):
#     print(f"Path: {output_path}")

#     files_list_checked = self.check_path(path, files_list, output_path)
#     if files_list_checked is None:
#         return
#     # data = nc.Dataset(self.file + self.path, 'r')
#     for file in files_list_checked:
#         full_path = os.path.join(path, file)
#         if not os.path.isfile(full_path):
#             print(f"Erreur : Le fichier '{file}' est introuvable dans '{path}'")
#             return
#         if not os.access(full_path, os.R_OK):
#             print(f"Erreur: Permission refusée pour accéder à '{file}' dans '{path}'")
#             return
#         if not file.endswith('.cdf'):
#             print(f"Erreur : Le fichier '{file}' n'est pas un fichier CDF valide.")
#             return
#         with nc.Dataset('mon_fichier.cdf', 'r') as dataset:
#             data_npy = {
#                 'scan_acquisition_time': dataset['scan_acquisition_time'][:],
#                 'mass_values': dataset['mass_values'][:],
#                 'intensity_values': dataset['intensity_values'][:],
#                 'total_intensity': dataset['total_intensity'][:],
#                 'point_count': dataset['point_count'][:],
#                 'mass_range_min': dataset['mass_range_min'][:],
#                 'mass_range_max': dataset['mass_range_max'][:],
#                 'scan_number': dataset.dimensions['scan_number'].size,
#             }

#         #save the data to a .npy file
#         base_name = f'{output_path}{file[:-4]}.npy'
#         np.save(base_name, data_npy)
#         print(f"Converted {file} to {base_name}")
# from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
import gc


class DataConverter:
    def __init__(self):
        self.default_path_input = os.getenv("HOST_VOLUME_PATH", "uploads")
        self.default_path_output = os.getenv("HOST_VOLUME_PATH", "converted_data")
    
    def get_files_from_folder(self, path):
        """Get all CDF files from a folder."""
        if os.path.isdir(path):
            return [f for f in os.listdir(path) if f.endswith(".cdf")]
        else:
            return []
    
    def check_path(self, path, files_list, output_path):
        """Check if the files exist and are readable."""
        messages = []
        
        if not os.path.isdir(path):
            messages.append(f"Erreur : Le chemin '{path}' n'est pas un répertoire valide.")
            return None, messages
            
        if not os.access(path, os.R_OK):
            messages.append(f"Erreur : Permission refusée pour accéder au répertoire '{path}'")
            return None, messages
            
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
                messages.append(f"Créé : {output_path}")
            except PermissionError:
                messages.append(f"Erreur : Permission refusée pour créer le répertoire '{output_path}'")
                return None, messages
        
        if not path:
            messages.append("Erreur : Aucun chemin sélectionné.")
            return None, messages
        
        if files_list is None:
            files_list = self.get_files_from_folder(path)
        
        messages.append(f"Files to convert: {files_list}")
        
        files_list = [file.strip() for file in files_list if file.strip()]
        messages.append(f"Fichiers à analyser : {files_list}")
        
        return files_list, messages
    
    def get_free_space(self, path):
        """Get free disk space in bytes."""
        import shutil
        try:
            return shutil.disk_usage(path).free
        except Exception as e:
            print(f"Erreur lors de la récupération de l'espace disque : {str(e)}")
            return float('inf')  # Si on ne peut pas vérifier, on continue
    
    def read_cdf_to_npy(self, path, files_list, output_path, progress_callback=None):
        """Convert CDF files to NPY format with memory optimization."""
        messages = []
        converted_files = []
        
        messages.append(f"Path: {output_path}")
        
        files_list_checked, check_messages = self.check_path(path, files_list, output_path)
        messages.extend(check_messages)
        
        if files_list_checked is None:
            return False, messages, []
        
        total_files = len(files_list_checked)
        
        for idx, file in enumerate(files_list_checked):
            try:
                full_path = os.path.join(path, file)
                
                if not os.path.isfile(full_path):
                    messages.append(f"Erreur : Le fichier '{file}' est introuvable dans '{path}'")
                    continue
                    
                if not os.access(full_path, os.R_OK):
                    messages.append(f"Erreur: Permission refusée pour accéder à '{file}' dans '{path}'")
                    continue
                    
                if not file.endswith('.cdf'):
                    messages.append(f"Erreur : Le fichier '{file}' n'est pas un fichier CDF valide.")
                    continue
                
                # Vérifier l'espace disque disponible
                file_size = os.path.getsize(full_path)
                free_space = self.get_free_space(output_path)
                
                if free_space < file_size * 2:  # Besoin d'au moins 2x la taille pour la conversion
                    messages.append(f"Erreur : Espace disque insuffisant pour {file} (besoin: {file_size*2//1024//1024}MB, disponible: {free_space//1024//1024}MB)")
                    continue
                
                messages.append(f"Conversion de {file} ({file_size//1024//1024}MB) - {idx+1}/{total_files}")
                
                if progress_callback:
                    progress_callback(idx, total_files, f"Traitement de {file}")
                
                # Lire le fichier CDF avec gestion mémoire optimisée
                with nc.Dataset(full_path, 'r') as dataset:
                    # Lire les données par chunks pour économiser la mémoire
                    data_npy = {}
                    
                    for var_name in ['scan_acquisition_time',
                                     'mass_values',
                                     'intensity_values',
                                     'total_intensity',
                                     'point_count',
                                     'mass_range_min',
                                     'mass_range_max']:
                    # for var_name in ['mass_values',
                    #                  'intensity_values']:
                        if var_name in dataset.variables:
                            var_data = dataset[var_name]
                            # Pour les gros arrays, on peut les traiter par chunks
                            if var_data.size > 10000000:  # > 10M éléments
                                messages.append(f"  - Lecture par chunks de {var_name} ({var_data.size} éléments)")
                                data_npy[var_name] = var_data[:]
                            else:
                                data_npy[var_name] = var_data[:]
                        else:
                            messages.append(f"  - Variable {var_name} non trouvée dans {file}")
                    
                    # Dimensions
                    if 'scan_number' in dataset.dimensions:
                        data_npy['scan_number'] = dataset.dimensions['scan_number'].size
                
                # Sauvegarder en .npy pour un accès rapide
                base_name = os.path.join(output_path, f'{file[:-4]}.npy')
                np.save(base_name, data_npy)
                
                # Nettoyer la mémoire
                del data_npy
                gc.collect()
                
                messages.append(f"✅ Converti {file} vers {base_name}")
                converted_files.append(base_name)
                
                if progress_callback:
                    progress_callback(idx+1, total_files, f"Terminé: {file}")
                
            except MemoryError:
                messages.append(f"❌ Erreur mémoire lors de la conversion de {file} - Fichier trop volumineux")
                gc.collect()
            except Exception as e:
                messages.append(f"❌ Erreur lors de la conversion de {file}: {str(e)}")
                gc.collect()
        
        return len(converted_files) > 0, messages, converted_files
    