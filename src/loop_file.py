import os, sys
path_to_scr_folder=os.path.join(os.path.dirname(os.path.abspath('')), 'src')
sys.path.append(path_to_scr_folder)
from identification import sample_identification
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':

    dir="D:/GCxGC/gamme/"
    files = [f for f in os.listdir(dir) if f.lower().endswith('.cdf')]
    for file in files:
        sample_identification(dir, file, dir + "output_als_2/",
                          mod_time=1.7,
                          method="DoG", mode="mass_per_mass",
                          noise_factor=5, abs_thresholds=500,
                          rel_thresholds=0.001,
                          cluster=0.5,
                          min_distance=1, min_sigma=1, max_sigma=3, sigma_ratio=1.5,
                          num_sigma=10,
                          formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
                          overlap=0.5, eps=0.001, min_samples=1, method_baseline="als",nist=False, 
                          quant_method="mass", extract_patch=False, output_hdf5_file=None, plot_=False)
    for file in files:
        sample_identification(dir, file, dir + "output_windows_2/",
                          mod_time=1.7,
                          method="DoG", mode="mass_per_mass",
                          noise_factor=5, abs_thresholds=500,
                          rel_thresholds=0.001,
                          cluster=0.5,
                          min_distance=1, min_sigma=1, max_sigma=3, sigma_ratio=1.5,
                          num_sigma=10,
                          formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
                          overlap=0.5, eps=0.001, min_samples=1, method_baseline="window",nist=False, 
                          quant_method="mass", extract_patch=False, output_hdf5_file=None, plot_=False)
    for file in files:
        sample_identification(dir, file, dir + "output_poly_2/",
                          mod_time=1.7,
                          method="DoG", mode="mass_per_mass",
                          noise_factor=5, abs_thresholds=500,
                          rel_thresholds=0.001,
                          cluster=0.5,
                          min_distance=1, min_sigma=1, max_sigma=3, sigma_ratio=1.5,
                          num_sigma=10,
                          formated_spectra=True, match_factor_min=600, min_persistence=0.0002,
                          overlap=0.5, eps=0.001, min_samples=1, method_baseline="poly",nist=False, 
                          quant_method="mass", extract_patch=False, output_hdf5_file=None, plot_=False)
        

        

        