import os
import numpy as np
import netCDF4 as nc
# from datetime import datetime
import gc
import os
import time
import numpy as np
import netCDF4 as nc
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class DataConverter:
    def __init__(self):
        self.default_path_input = os.getenv("HOST_VOLUME_PATH")
        self.default_path_output = os.getenv("HOST_VOLUME_PATH")
        self.progress_lock = threading.Lock()
        self.completed = 0

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
            return float('inf')

    def write_var_to_hdf5(self, nc_dataset, h5_file, var_name):
        """Écrit une variable NetCDF dans un fichier HDF5 avec conversion de type et compression."""
        try:
            if var_name in nc_dataset.variables:
                data = nc_dataset[var_name][:]
                if data.dtype == np.float64 and var_name != 'intensity_values':
                    data = data.astype(np.float32)

                h5_file.create_dataset(var_name,
                                       data=data,
                                       compression='lzf')
                del data
            else:
                print(f"⚠️ Variable {var_name} absente du fichier NetCDF")

        except Exception as e:
            print(f"❌ Erreur lors de l'écriture de {var_name} : {e}")

    def verify_converted_file(self, hdf5_path, file_name=None):
        """Vérifier l'intégrité du fichier H5 converti."""
        try:
            with h5py.File(hdf5_path, 'r') as h5f:
                # Vérifier que toutes les variables existent
                required_vars = ['scan_acquisition_time', 'mass_values',
                                 'intensity_values', 'total_intensity',
                                 'point_count']
                
                for var in required_vars:
                    if var not in h5f:
                        error_msg = f"Variable {var} manquante"
                        if file_name:
                            error_msg = f"[{file_name}] {error_msg}"
                        return False, error_msg
                
                # Vérifier la cohérence des tailles
                point_count = h5f['point_count'][:]
                expected_size = point_count.sum()
                actual_mass_size = h5f['mass_values'].size
                actual_intensity_size = h5f['intensity_values'].size
                
                if actual_mass_size != expected_size:
                    error_msg = f"Incohérence tailles: {actual_mass_size} vs {expected_size}"
                    if file_name:
                        error_msg = f"[{file_name}] {error_msg}"
                    return False, error_msg

                if actual_intensity_size != expected_size:
                    error_msg = f"Incohérence intensity_values: {actual_intensity_size} vs {expected_size} attendu"
                    if file_name:
                        error_msg = f"[{file_name}] {error_msg}"
                    return False, error_msg

                return True, "OK"

        except Exception as e:
            error_msg = f"Erreur lecture: {e}"
            if file_name:
                error_msg = f"[{file_name}] {error_msg}"
            return False, error_msg

    def convert_single_file_optimized(self, file_info):
        """Convert a single CDF file to HDF5 with float32 optimization."""
        full_path, file_name, output_path, file_idx, total_files = file_info
        messages = []

        try:
            hdf5_path = os.path.join(output_path, f'{file_name[:-4]}.h5')
            if os.path.exists(hdf5_path):
                print(f"Le fichier {hdf5_path} existe déjà. Vérification...")
                h5_file_size = os.path.getsize(hdf5_path)

                if h5_file_size > 0:  # Fichier non vide
                    # Vérifier l'intégrité
                    is_valid, integrity_msg = self.verify_converted_file(hdf5_path, file_name)
                    if is_valid:
                        messages.append(f"ℹ️ [{file_idx+1}/{total_files}] {file_name} - Déjà converti")
                        return True, messages, hdf5_path
                    else:
                        messages.append(f"⚠️ Fichier corrompu détecté: {integrity_msg}")
                        os.remove(hdf5_path)
                        messages.append(f"🗑️ Fichier corrompu supprimé, reconversion...")
                else:
                    # Fichier existe mais est vide, le supprimer
                    os.remove(hdf5_path)
                    messages.append(f"🗑️ Fichier vide supprimé: {file_name[:-4]}.h5")

            # Vérifier l'espace disque disponible
            cdf_file_size = os.path.getsize(full_path)
            if cdf_file_size == 0:
                messages.append(f"❌ Fichier vide: {file_name}")
                return False, messages, None
            
            free_space = self.get_free_space(output_path) 
            if free_space < cdf_file_size * 2:  # Besoin d'au moins 2x la taille pour la conversion
                messages.append(f"Erreur : Espace disque insuffisant pour {file_name} (besoin: {cdf_file_size*2//1024//1024}MB, disponible: {free_space//1024//1024}MB)")
                return False, messages, None

            start_time = time.time()

            # Lire le fichier CDF avec gestion mémoire optimisée
            with nc.Dataset(full_path, 'r', encoding="latin-1") as dataset:
                with h5py.File(hdf5_path, 'w') as h5f:
                    # Conversion mass_values en float32
                    for var in ['scan_acquisition_time',
                                'mass_values',
                                'intensity_values',
                                'total_intensity',
                                'point_count',
                                'mass_range_min',
                                'mass_range_max']:
                        self.write_var_to_hdf5(dataset, h5f, var)

                    if 'scan_number' in dataset.dimensions:
                        size = dataset.dimensions['scan_number'].size
                        h5f.attrs['scan_number_size'] = size

            # Vérifier après conversion
            is_valid, integrity_msg = self.verify_converted_file(hdf5_path, file_name)
            if not is_valid:
                messages.append(f"❌ Fichier converti corrompu: {integrity_msg}")
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)
                return False, messages, None

            gc.collect()
            conversion_time = time.time() - start_time
            output_size_mb = os.path.getsize(hdf5_path) // 1024 // 1024
            messages.append(f"✅ [{file_idx+1}/{total_files}] {file_name} terminé en {conversion_time:.1f}s")
            messages.append(f"   📦 Taille: {cdf_file_size // 1024 // 1024}MB → {output_size_mb}MB") # (compression {compression_ratio:.1f}x)")
            return True, messages, hdf5_path

        except MemoryError:
            messages.append(f"❌ Erreur mémoire pour {file_name}")
            gc.collect()
            return False, messages, None
        except Exception as e:
            messages.append(f"❌ Erreur conversion {file_name}: {str(e)}")
            return False, messages, None

    def get_max_workers(self, files):
        cpu_count = os.cpu_count() or 1
        max_allowed = min(2 * cpu_count, 8)
        return min(len(files), max_allowed)

    def convert_cdf_to_hdf5_threaded(self, path, files_list, output_path):
        """Convert CDF files to HDF5 with float32 optimization and with
        threading."""
        messages = []
        converted_files = []
        self.completed = 0

        messages.append(f"🚀 Conversion avec HDF5 + Float32")
        messages.append(f"📁 Dossier source: {path}")
        messages.append(f"📁 Dossier sortie: {output_path}")
        files_list_checked, check_messages = self.check_path(path, files_list, output_path)
        messages.extend(check_messages)

        if files_list_checked is None:
            return False, messages, []

        max_workers = self.get_max_workers(files_list_checked)
        messages.append(f"👥 Workers: {max_workers}")

        valid_files = []
        for file in files_list_checked:
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK) and file.endswith('.cdf'):
                valid_files.append(file)
            else:
                messages.append(f"Erreur : Le fichier '{file}' est introuvable ou n'est pas accessible dans '{path}'")
        if not valid_files:
            return False, messages + ["❌ Aucun fichier CDF valide trouvé"], []
        total_files = len(valid_files)
        # Préparation des tâches
        file_infos = [
            (os.path.join(path, file), file, output_path, idx, total_files)
            for idx, file in enumerate(valid_files)
        ]
        start_total = time.time()
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="CDFConverter") as executor:
            future_to_info = {
                executor.submit(self.convert_single_file_optimized, info): info
                for info in file_infos
            }
            for future in as_completed(future_to_info):
                file_info = future_to_info[future]
                file_name = file_info[1]

                try:
                    success, file_messages, converted_file = future.result()
                    with self.progress_lock:
                        messages.extend(file_messages)
                        self.completed += 1
                        if success and converted_file:
                            converted_files.append(converted_file)
                except Exception as e:
                    with self.progress_lock:
                        messages.append(f"❌ Erreur thread pour {file_name}: {str(e)}")
                        self.completed += 1

        total_time = time.time() - start_total

        messages.append(f"\n📈 RÉSULTATS:")
        messages.append(f"Fichiers convertis: {len(converted_files)}/{total_files}")
        messages.append(f"⏱️  Temps total: {total_time:.1f}s")
        messages.append(f"⚡ Temps moyen/fichier: {total_time/total_files:.1f}s")
        return len(converted_files) > 0, messages, converted_files
