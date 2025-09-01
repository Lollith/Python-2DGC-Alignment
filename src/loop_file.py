import os, sys
path_to_scr_folder=os.path.join(os.path.dirname(os.path.abspath('')), 'src')
sys.path.append(path_to_scr_folder)
from identification import sample_identification
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':

    # dir='//papillon.sssv.uvsq.fr/spectro/Etudiants/Camille'
    # files = [f for f in os.listdir(dir) if f.lower().endswith('.cdf')]
    # for file in files[9:]:
    #     sample_identification(dir, file, dir + "/output/",
    #                         mod_time=1.7,
    #                         method="DoG", mode="mass_per_mass",
    #                         noise_factor=3, abs_thresholds=500,
    #                         rel_thresholds=0.0001,
    #                         cluster=0.5,
    #                         min_distance=1, min_sigma=1, max_sigma=20, sigma_ratio=2,
    #                         num_sigma=10,
    #                         formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
    #                         overlap=0.5, eps=0.001, min_samples=1, nist=False,
    #                         quant_method="mass", extract_patch=False,output_hdf5_file= "", 
    #                         plot_=False)
    # dir="D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Manue/GCxGC_VOLATIL-CF_02/"
    # files = [f for f in os.listdir(dir) if f.lower().endswith('.cdf')]
    # for file in files:
    #     sample_identification(dir, file, dir + "output_mass_par_mass/",
    #                         mod_time=1.7,
    #                         method="DoG", mode="mass_per_mass",
    #                         noise_factor=3, abs_thresholds=1000,
    #                         rel_thresholds=0.001,
    #                         cluster=0.5,
    #                         min_distance=1, min_sigma=1, max_sigma=20, sigma_ratio=2,
    #                         num_sigma=10,
    #                         formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
    #                         overlap=0.5, eps=0.001, min_samples=1, nist=False,
    #                         quant_method="mass", extract_patch=True,
    #                         output_hdf5_file= dir + "output_mass_par_mass/dataset.h5", 
    #                         plot_=False)
        

    dir="D:/GCxGC_MS/DATA/Dossier_partagé_GCxGC/Elo/APROCCHSS_sequence septembre 2023/CDF/"
    files = [f for f in os.listdir(dir) if f.lower().endswith('.cdf')]
    for file in files:
        sample_identification(dir, file, dir + "output_mass_par_mass_dog/",
                            mod_time=1.25,
                            method="DoG", mode="mass_per_mass",
                            noise_factor=5, abs_thresholds=1000,
                            rel_thresholds=0.005,
                            cluster=0.5,
                            min_distance=1, min_sigma=1, max_sigma=10, sigma_ratio=1.1,
                            num_sigma=10,
                            formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
                            overlap=0.5, eps=0.001, min_samples=1, nist=False,
                            quant_method="mass", extract_patch=True,
                            output_hdf5_file= dir + "output_mass_par_mass_dog/dataset.h5", 
                            plot_=False)
        

        

        