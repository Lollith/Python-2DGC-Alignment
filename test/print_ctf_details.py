import netCDF4 as nc
import numpy as np  


fichier = '/media/camille/DATA1/cdf centroid/G0/G0-1-100123.cdf'

def print_ctf_details():
    try:
        data = nc.Dataset(fichier, 'r')
        print("📂 Dimensions disponibles :", list(data.dimensions.keys()))
        print("📊 Variables disponibles :", list(data.variables.keys()))

        # Afficher les attributs globaux du fichier
        print("\n📑 Attributs globaux :")
        for attr in data.ncattrs():
            print(f"{attr} : {data.getncattr(attr)}")

        print("\n🧐 Détails des variables :")
        for var in data.variables.values():
            print(f"Nom : {var.name}")
            print(f"Dimensions : {var.dimensions}")
            print(f"Taille : {var.size}")
            print(f"Type de données : {var.dtype}")
            print(f"Attributs : {[attr for attr in var.ncattrs()]}")
            print("----\n")

        data.close()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")


def print_intensity():
    data = nc.Dataset(fichier, 'r')
    chromato = data['total_intensity']
    time_rn = data['scan_acquisition_time']
    mass_values = data['mass_values']
    intensity_values = data['intensity_values']
    mass_range_min = data['mass_range_min']
    mass_range_max = data['mass_range_max']
    point_count = data['point_count']
    if point_count.dtype.kind in 'O':
        print("point count is object")
    else:
        print("point count is not object")

    print("total_intensity", chromato[:10])
    print("acquisition time", time_rn[:10])
    print("mass", mass_values[:10])
    print("intensity",intensity_values[:10])
    print("mass range min", mass_range_min[:10])
    print("mass range max", mass_range_max[:10])
    print("point count", point_count[:10])


def check_and_convert_point_count(filename):
    ds = nc.Dataset(filename)
    print("type", ds["point_count"].dtype)
    donnee = np.abs(ds["point_count"])
    mydonnee = np.abs([245, 227, 222, 218])< np.iinfo(np.int32).max
    print("donnee", donnee)
    print("mydonnee", mydonnee)
    

if __name__ == "__main__":
    print_ctf_details()
    print_intensity()
    check_and_convert_point_count(filename=fichier)