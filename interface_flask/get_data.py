import netCDF4 as nc
import os
import numpy as np


def read_cdf_to_npy(self, path, files_list, output_path):
    print(f"Path: {output_path}")

    files_list_checked = self.check_path(path, files_list, output_path)
    if files_list_checked is None:
        return
    # data = nc.Dataset(self.file + self.path, 'r')
    for file in files_list_checked:
        full_path = os.path.join(path, file)
        if not os.path.isfile(full_path):
            print(f"Erreur : Le fichier '{file}' est introuvable dans '{path}'")
            return
        if not os.access(full_path, os.R_OK):
            print(f"Erreur: Permission refusée pour accéder à '{file}' dans '{path}'")
            return
        if not file.endswith('.cdf'):
            print(f"Erreur : Le fichier '{file}' n'est pas un fichier CDF valide.")
            return
        with nc.Dataset('mon_fichier.cdf', 'r') as dataset:
            data_npy = {
                'scan_acquisition_time': dataset['scan_acquisition_time'][:],
                'mass_values': dataset['mass_values'][:],
                'intensity_values': dataset['intensity_values'][:],
                'total_intensity': dataset['total_intensity'][:],
                'point_count': dataset['point_count'][:],
                'mass_range_min': dataset['mass_range_min'][:],
                'mass_range_max': dataset['mass_range_max'][:],
                'scan_number': dataset.dimensions['scan_number'].size,
            }

        #save the data to a .npy file
        base_name = f'{output_path}{file[:-4]}.npy'
        np.save(base_name, data_npy)
        print(f"Converted {file} to {base_name}")
        