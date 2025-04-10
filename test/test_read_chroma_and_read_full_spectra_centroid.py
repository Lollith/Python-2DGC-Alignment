import read_chroma
import pandas as pd
import mass_spec
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    filename = '/media/camille/DATA1/cdf centroid/G0/G0-1-120123.cdf'


    print('---Check read_chroma---')
    chromato_obj = read_chroma.read_chroma(filename, 1.25)
    tic_chromato, time_rn, spectra_obj = chromato_obj
    print("tic: cf chromato.csv")
    df = pd.DataFrame(tic_chromato)
    df.to_csv("test/chromato.csv", index=False)

    print("duree de chromato: cf chromato1.csv")
    df1 = pd.DataFrame(time_rn)
    df1.to_csv("test/chromato1.csv", index=False)

    print("spectre de masse:")
    print("spectra l1", spectra_obj[0])
    print("spectra l2", spectra_obj[1])
    print("spectra masse value", spectra_obj[2])
    print("spectra intensity", spectra_obj[3])
    print("spectra range min", spectra_obj[4])
    print("spectra range max", spectra_obj[5])

    plt.imshow(tic_chromato, aspect='auto', cmap='hot', origin='lower')
    plt.title("TIC from read_chroma")
    plt.xlabel("RT2 (fast)")
    plt.ylabel("RT1 (slow)")
    plt.colorbar(label="Intensity")
    plt.savefig("test/tic_chromato.png")
    print("TIC chromatogram: cf tic_chromato.png")
    #TODO a comparer avec le resulta de chromaTof

    print("---------------------------------")
    mass_values = spectra_obj[2]
    int_values = spectra_obj[3]
    l1 = spectra_obj[0]
    l2 = spectra_obj[1]
    

    print('---Check read_full_spectra_centroid---')
    full_nominal_mass_spectra = mass_spec.read_full_spectra_centroid(spectra_obj)
    spectra_full_nom, debuts, fins = full_nominal_mass_spectra
    print("full_nom", spectra_full_nom)
    print("debuts et fins", debuts, fins)

    index = 1000
    mass_values, intensities = spectra_full_nom[index]

    plt.figure(figsize=(10, 5))
    plt.stem(mass_values, intensities)
    plt.title(f"Spectrum at index {index}")
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.savefig("test/spectre_index_1000.png")
    plt.show()

    #spectre moyen
    mass_values = spectra_full_nom[0][0]
    all_intensities = np.stack([s[1] for s in spectra_full_nom])
    mean_spectrum = np.mean(all_intensities, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(mass_values, mean_spectrum)
    plt.title("Spectre moyen (toutes acquisitions)")
    plt.xlabel("m/z")
    plt.ylabel("Intensité moyenne")
    plt.savefig("test/spectre_moyen.png")
    plt.show()



# seuil=5
# MIN_SEUIL = seuil * sigma * 100 / np.max(chromato)
"""
Calculates a dynamic minimum threshold (`MIN_SEUIL`) for detecting significant peaks in a chromatogram.

The threshold is based on the following parameters:
- `seuil` : A scaling factor that amplifies the threshold value (e.g., 5).
- `sigma` : The standard deviation or noise level in the chromatogram data.
- `chromato` : An array of chromatographic intensity values.

This calculation normalizes the threshold relative to the maximum intensity value in the chromatogram. The formula ensures that the threshold adapts based on both the noise level (`sigma`) and the overall intensity of the signal. The resulting threshold helps filter out insignificant noise and ensures that small peaks in low-intensity signals are not missed.

Parameters
----------
seuil : float
    A factor that scales the threshold.
sigma : float
    The noise level or standard deviation of the chromatogram.
chromato : ndarray
    The array containing the chromatogram intensity values.

Returns
-------
float
    The dynamic minimum threshold for detecting significant peaks, adjusted based on signal intensity and noise.
"""