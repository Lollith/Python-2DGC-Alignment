# import read_chroma
# import h5py
import numpy as np
import skimage.feature
import peak_detection

method = "peak_local_max"
mode = "tic"
import read_chroma
# file_path = '/home/camille/Documents/app/data/A-F-028-817822-droite-ReCIVA.h5'


abs_threshold = 0
rel_threshold = 0.01
noise_factor = 1.5
min_persistence = 0.02
        
min_distance = 1
sigma_ratio = 1.6
num_sigma = 10
min_sigma = 1
max_sigma = 30
overlap = 0.5
match_factor_min = 650
cluster = True
min_samples = 4
eps = 3



# skimage.feature.peak_local_max---------------------------------
def test_skimage_peak_local_max_basic():
    chromato = np.array([
        [0, 0, 5, 0, 0],
        [0, 1, 0, 1, 0],
        [5, 0, 10, 0, 5],
        [0, 1, 0, 1, 0],
        [0, 0, 5, 0, 0]]
        )

    coordinates = skimage.feature.peak_local_max(
        image=chromato,
        min_distance=min_distance,
        threshold_abs=abs_threshold,
    )

    expected = np.array([[2, 2]])

    np.testing.assert_array_equal(coordinates, expected)


def test_skimage_peak_local_max_multiple():
    image = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 8, 0], 
        [0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 12, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 9, 0],
        [0, 0, 0, 0, 0, 0, 0]  
    ])

    coordinates = skimage.feature.peak_local_max(
        image, min_distance=min_distance, threshold_abs=abs_threshold)
    expected = np.array([
        [1, 1],
        [1, 5],
        [3, 3],
        [5, 1],
        [5, 5]
    ])
    assert set(map(tuple, coordinates)) == set(map(tuple, expected))


def test_skimage_tic_peak_local_max_chromato():
    chromato_tic = np.array([
        10, 12, 14, 18, 23, 45, 90, 45, 23, 15, 12, 10, 11, 30, 70, 30, 12, 11, 10, 9
    ])
    # Pour un signal 1D, reshape en 2D si nécessaire
    coordinates = skimage.feature.peak_local_max(chromato_tic, min_distance=2, threshold_abs=20)
    detected_peaks = coordinates[:, 0]
    print("Coordinates:", coordinates)
    expected = np.array([6, 14])  # indices des pics principaux
    assert set(detected_peaks) == set(expected), f"Expected {expected} but got {detected_peaks}"


def test_skimage_tic_peak_local_max_with_noise():
    chromato_tic = np.array([
        10, 12, 14, 18, 23, 45, 90, 45, 23, 15, 12, 10, 11, 30, 70, 30, 12, 11, 10, 9
    ]) + np.random.normal(0, 5, size=20)  # Ajout de bruit
    print("Chromato TIC with noise:", chromato_tic)

    coordinates = skimage.feature.peak_local_max(chromato_tic, min_distance=2, threshold_abs=20)
    detected_peaks = coordinates[:, 0]
    # print("Coordinates with noise:", coordinates)
    expected = np.array([6, 14])  # indices des pics principaux
    assert set(detected_peaks) == set(expected), f"Expected {expected} but got {detected_peaks}"

def test_skimage_tic_peak_local_max_with_low_threshold():
    chromato_tic = np.array([
        10, 12, 14, 18, 23, 45, 90, 45, 23, 15, 12, 10, 11, 30, 70, 30, 12, 11, 10, 9
    ])
    coordinates = skimage.feature.peak_local_max(chromato_tic, min_distance=2, threshold_abs=0)
    detected_peaks = coordinates[:, 0]
    expected = np.array([6, 14])  # indices des pics principaux
    assert set(detected_peaks) == set(expected), f"Expected {expected} but got {detected_peaks}"

#

def test_tic_peak_detection_with_noise():
    chromato_tic = np.array([
        10, 12, 14, 18, 23, 45, 90, 45, 23, 15, 12, 10, 11, 30, 70, 30, 12, 11, 10, 9
    ]) + np.random.normal(0, 5, size=20)  # Ajout de bruit
    print("Chromato TIC with noise:", chromato_tic)
    chromato_tic, time_rn, chromato_cube, sigma, mass_range = (
        read_chroma.read_chromato_and_chromato_cube(chromato_tic, mod_time=1,
                                                    pre_process=True
                                                    ))

    coordinates = peak_detection.peak_detection(chromato_tic, time_rn=None, mass_range=None),
        chromato_cube=chromato_cube,
        sigma=None,
        noise_factor=noise_factor,
        abs_threshold=abs_threshold,
        rel_threshold=rel_threshold,
        method=method,
        mode=mode,
        cluster=cluster,
        min_distance=min_distance,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        num_sigma=num_sigma,
        min_persistence=min_persistence,
        overlap=overlap,
        eps=eps,
        min_samples=min_samples)
    detected_peaks = coordinates[:, 0]
    # print("Coordinates with noise:", coordinates)
    expected = np.array([6, 14])  # indices des pics principaux
    assert set(detected_peaks) == set(expected), f"Expected {expected} but got {detected_peaks}"

