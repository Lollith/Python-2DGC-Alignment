import read_chroma
import pandas as pd
import mass_spec

if __name__ == "__main__":
    filename = '/media/camille/DATA1/cdf centroid/G0/G0-1-100123.cdf'

    #TODO : recuperer des .cdf de standard + pytest

    print('Check read_chroma')
    chromato_obj = read_chroma.read_chroma(filename, 1.25)
    chromato, time_rn, spectra_obj = chromato_obj
    print("tic: cf chromato.csv")
    df = pd.DataFrame(chromato)
    df.to_csv("chromato.csv", index=False)

    print("duree de chromato: cf chromato1.csv")
    df1 = pd.DataFrame(time_rn)
    df1.to_csv("chromato1.csv", index=False)

    print("spectre de masse:")
    print("spectra l1", spectra_obj[0])
    print("spectra l2", spectra_obj[1])
    print("spectra masse value", spectra_obj[2])
    print("spectra intensity", spectra_obj[3])
    print("spectra range min", spectra_obj[4])
    print("spectra range max", spectra_obj[5])
    print("---------------------------------")

mass_values = spectra_obj[2]
int_values = spectra_obj[3]


# print('Check centroid_to_full_nominal')
# intensity_v = mass_spec.centroid_to_full_nominal(spectra_obj, mass_values, int_values)
# print("intensity", intensity_v)

print('Check read_full_spectra_centroid')
full_nominal_mass_spectra = mass_spec.read_full_spectra_centroid(spectra_obj)
print(full_nominal_mass_spectra)


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