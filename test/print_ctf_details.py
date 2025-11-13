import netCDF4 as nc
import numpy as np
import os
from datetime import datetime
import sys


# fichier = '/home/camille/Documents/app/data/input/G0/G0-1-120123.cdf'
#fichier = '/home/camille/Documents/app/data/cdf et h5/J-A-034-751325-Tedlar.cdf'
fichier = 'd:/GCxGC_MS/DATA/testgamme/cdff/864684_gamme_1ppm.cdf'
# fichier = '/media/camille/DATA2/cdf centroid/817831-blanc-piece-fin-210823.cdf'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def print_ctf_details():
    try:
        print(f"\n🔍 Fichier ouvert : {fichier}\n")
        data = nc.Dataset(fichier, 'r')

        # Dimensions
        print("📏 Dimensions disponibles :")
        for dim_name, dim in data.dimensions.items():
            print(f"- {dim_name} : taille = {len(dim)}")
        print()

        # Variables
        print("📊 Variables disponibles :")
        for var_name in data.variables:
            print(f"- {var_name}")
        print()

        # Attributs globaux
        print("📑 Attributs globaux :")
        for attr in data.ncattrs():
            print(f"- {attr} : {data.getncattr(attr)}")
            if 'date' in attr.lower() or 'time' in attr.lower():
                    print()
                    print(f"   {attr}: {data.getncattr(attr)}")
        print()

        # Détails des variables
        print("🧐 Détails des variables :")
        for var_name, var in data.variables.items():
            print(f"🔸 Nom : {var.name}")
            print(f"   ↪ Dimensions : {var.dimensions}")
            print(f"   ↪ Taille : {var.size}")
            print(f"   ↪ Type de données : {var.dtype}")
            print(f"   ↪ Attributs : {[attr for attr in var.ncattrs()]}")
            print(f"   scan_acquisition_time: {data['scan_acquisition_time'][:10]}")
            try:
                print(f"   ↪ Valeurs (extrait) : {var[:10]}")
            except:
                print("   ⚠ Impossible d'afficher les valeurs (trop de dimensions ?)")
            print("----")

        # Vérification spécifique : point_count
        print("\n🔍 Vérification de la variable 'point_count'...")
        if 'point_count' in data.variables:
            print("✔️ 'point_count' est bien une variable.")
            try:
                print("   ↪ Valeurs : ", data['point_count'][:10])
            except Exception as e:
                print(f"   ⚠ Impossible d'afficher les valeurs : {e}")
        elif 'point_count' in data:
            print("❓ 'point_count' est accessible dans data mais pas dans data.variables.")
            try:
                print("   ↪ Valeurs : ", data['point_count'][:10])
            except Exception as e:
                print(f"   ⚠ Impossible d'afficher les valeurs : {e}")
        else:
            print("❌ 'point_count' n'existe pas dans ce fichier.")

        mtime = os.path.getmtime(fichier)
        print(f"\n💾 Date modification fichier: {datetime.fromtimestamp(mtime)}")

        data.close()

    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {e}")


def print_intensity():
    data = nc.Dataset(fichier, 'r')
    chromato = data['total_intensity']
    time_rn = data['scan_acquisition_time']
    mass_values = data['mass_values']
    intensity_values = data['intensity_values']
    mass_range_min = data['mass_range_min']
    mass_range_max = data['mass_range_max']
    point_count = data['point_count']
    scan_duration = data['scan_duration']
    # mod_time = data['mod_time']

    print("total_intensity", chromato[:10])
    print("acquisition time", time_rn[:10])
    print("mass", mass_values[:10])
    print("intensity",intensity_values[:10])
    print("mass range min", mass_range_min[:10])
    print("mass range max", mass_range_max[:10])
    print("point count", point_count[:])
    print("scan duration", scan_duration[:10])
    # print("mod time", mod_time[:10])

    if 'mod_time' in data.ncattrs():
        mod_time = data.getncattr('mod_time')  # Extraire la valeur du mod_time
        print(f"Modulation Time (mod_time) : {mod_time}")
    else:
        print("mod_time non trouvé dans les métadonnées.")

def check_and_convert_point_count(filename):
    ds = nc.Dataset(filename)
    print("type", ds["point_count"].dtype)
    donnee = np.abs(ds["point_count"])
    mydonnee = np.abs([245, 227, 222, 218])< np.iinfo(np.int32).max
    print("donnee", donnee)
    print("mydonnee", mydonnee)
    

if __name__ == "__main__":
    print_ctf_details()
    # print_intensity()
    # check_and_convert_point_count(filename=fichier)