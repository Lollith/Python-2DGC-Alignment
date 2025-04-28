from plot import visualizer
# from skimage.feature import peak_local_max
import skimage.feature
from scipy import ndimage as ndi
from imagepers import persistence
from skimage.feature import blob_dog, blob_log, blob_doh
import math
import numpy as np
# import pywt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import multiprocessing
# from multiprocessing import Pool

# # lissage
# def gaussian_filter(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a Gaussian filter to the chromatogram for smoothing.

#     This function uses a Gaussian filter to smooth the chromatogram values, which can help reduce noise in the data.
#     After applying the filter, the smoothed chromatogram is visualized, and the result of the `peak_local_max` function for peak detection is returned.

#     Parameters:
#     chromato_obj : tuple
#         The chromatogram as a matrix and the associated time.
#     mod_time : float
#         The modulation time.
#     seuil : float, optional
#         The relative threshold for detecting local peaks.
#     threshold_abs : float, optional
#         The absolute threshold for filtering the minimum peak intensities.

#     Returns:
#     array
#         The coordinates of the peaks detected by the `peak_local_max` function.

#     Visualization:
#         The filtered chromatogram is visualized.
#     """

#     chromato, time_rn = chromato_obj
#     sobel_chromato = ndi.gaussian_filter(chromato, sigma=5)
#     visualizer((sobel_chromato, time_rn), mod_time, title="Chromato Gauss Filter")
#     return ((sobel_chromato, time_rn), mod_time, seuil)

# # detection contours
# def gauss_laplace(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a Gaussian-Laplacian filter to the chromatogram for edge detection.

#     This filter is used to detect edges in the chromatogram by combining a Gaussian filter and a Laplacian operator.
#     After applying the filter, the processed chromatogram is visualized, and the result of the `peak_local_max` function for peak detection is returned.

#     Parameters:
#     chromato_obj : tuple
#         The chromatogram as a matrix and the associated time.
#     mod_time : float
#         The modulation time.
#     seuil : float, optional
#         The relative threshold for detecting local peaks.
#     threshold_abs : float, optional
#         The absolute threshold for filtering the minimum peak intensities.

#     Returns:
#     array
#         The coordinates of the peaks detected by the `peak_local_max` function.

#     Visualization:
#         The filtered chromatogram is visualized.
#     """

#     chromato, time_rn = chromato_obj
#     sobel_chromato = ndi.gaussian_laplace(chromato, sigma=5)
#     visualizer((sobel_chromato, time_rn), mod_time, title="Chromato Gauss Laplace")
#     return peak_local_max((sobel_chromato, time_rn), mod_time, seuil)

# # detection contours
# def gauss_multi_deriv(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a Gaussian multi-derivative gradient filter to the chromatogram for edge detection.

#     This filter calculates successive derivatives of the chromatogram to highlight edges.
#     After applying the filter, the processed chromatogram is visualized, and the result of the `peak_local_max` function for peak detection is returned.

#     Parameters:
#     chromato_obj : tuple
#         The chromatogram as a matrix and the associated time.
#     mod_time : float
#         The modulation time.
#     seuil : float, optional
#         The relative threshold for detecting local peaks.
#     threshold_abs : float, optional
#         The absolute threshold for filtering the minimum peak intensities.

#     Returns:
#     array
#         The coordinates of the peaks detected by the `peak_local_max` function.

#     Visualization:
#         The filtered chromatogram is visualized.
#     """
        
#     chromato, time_rn = chromato_obj
#     sobel_chromato = ndi.gaussian_gradient_magnitude(chromato, sigma=5)
#     visualizer((sobel_chromato, time_rn), mod_time, title="Chromato Gauss Mutli Deriv")
#     return peak_local_max((sobel_chromato, time_rn), mod_time, seuil)

# # detection contours
# def prewitt(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a Prewitt filter to the chromatogram for edge detection.

#     The Prewitt filter is an edge-detection operator used to highlight the contours in the chromatogram.
#     After applying the filter, the processed chromatogram is visualized, and the result of the `peak_local_max` function for peak detection is returned.

#     Parameters:
#     chromato_obj : tuple
#         The chromatogram as a matrix and the associated time.
#     mod_time : float
#         The modulation time.
#     seuil : float, optional
#         The relative threshold for detecting local peaks.
#     threshold_abs : float, optional
#         The absolute threshold for filtering the minimum peak intensities.

#     Returns:
#     array
#         The coordinates of the peaks detected by the `peak_local_max` function.

#     Visualization:
#         The filtered chromatogram is visualized.
#     """

#     chromato, time_rn = chromato_obj
#     sobel_chromato = ndi.prewitt(chromato)
#     visualizer((sobel_chromato, time_rn), mod_time, title="Chromato Prewitt")
#     return peak_local_max((sobel_chromato, time_rn), mod_time, seuil)

# # detection contours
# def sobel(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a Sobel filter to the chromatogram for edge detection.

#     The Sobel filter is another edge-detection operator. After applying this filter, 
#     the processed chromatogram is visualized, and the result of the `peak_local_max` function for peak detection is returned.

#     Parameters:
#     chromato_obj : tuple
#         The chromatogram as a matrix and the associated time.
#     mod_time : float
#         The modulation time.
#     seuil : float, optional
#         The relative threshold for detecting local peaks.
#     threshold_abs : float, optional
#         The absolute threshold for filtering the minimum peak intensities.

#     Returns:
#     array
#         The coordinates of the peaks detected by the `peak_local_max` function.

#     Visualization:
#         The filtered chromatogram is visualized.
#     """

#     chromato, time_rn = chromato_obj
#     sobel_chromato = ndi.sobel(chromato)
#     visualizer((sobel_chromato, time_rn), mod_time, title="Chromato Sobel")
#     return peak_local_max((sobel_chromato, time_rn), mod_time, seuil)

# def tf(chromato_obj, mod_time, seuil):
#     """
#     Applies a 2D Fourier Transform (FFT 2D) to a chromatogram and displays its frequency spectrum.

#     This function helps analyze periodicities in the chromatographic signal, 
#     particularly those related to the modulation time in GC×GC-MS. It can be used to:
#     - Identify recurring patterns in the chromatogram.
#     - Detect instrumental artifacts or periodic noise.
#     - Improve signal alignment and correction in chromatographic analysis.

#     Parameters:
#     -----------
#     chromato_obj : tuple (numpy.ndarray, any)
#         - `chromato`: 2D matrix representing the chromatogram (intensity as a function of time and mass).
#         - `time_rn`: (Not used here) May represent an associated time vector.

#     mod_time : float
#         Modulation time used in two-dimensional gas chromatography.

#     seuil : float
#         Threshold for potential frequency filtering.

#     Returns:
#     --------
#     None
#         Displays an image of the frequency spectrum after applying the 2D FFT.

#     Example:
#     --------
#     >>> tf((chromato_matrix, time_vector), mod_time=5.0, seuil=0.1)
#     (Displays the frequency spectrum of the chromatogram)
#     """
#     chromato, time_rn = chromato_obj
#     fft_MTBLS08 = np.fft.fft2(chromato)
#     plt.imshow(np.abs(np.fft.fftshift(fft_MTBLS08)))
#     plt.show()

# def wavelet(chromato_obj, mod_time, seuil=0, threshold_abs=None):
#     """
#     Applies a 2D Discrete Wavelet Transform (DWT) to a chromatogram to decompose it into different frequency components. 
#     This function uses the 'bior1.3' biorthogonal wavelet filter. It visualizes the results by displaying the 
#     approximation (LL) and detail coefficients (LH, HL, HH) as contour plots.

#     Args:
#         chromato_obj (tuple): A tuple containing the chromatogram matrix and corresponding time values.
#         mod_time (str): A label or identifier used for visualizing the data.
#         seuil (float, optional): The threshold for peak detection. Default is 0.
#         threshold_abs (float, optional): An absolute threshold for peak detection. Default is None.

#     Returns:
#         numpy.ndarray: The coordinates of detected peaks after applying the `peak_local_max` function on the diagonal detail coefficients (HH).
        
#     Visualization:
#         This function visualizes four components from the 2D wavelet transform:
#         1. Approximation (LL)
#         2. Horizontal Detail (LH)
#         3. Vertical Detail (HL)
#         4. Diagonal Detail (HH)
#         Each component is displayed as a contour plot with an appropriate title.

#     Example:
#         wavelet(chromato_obj, mod_time='sample_mod', seuil=0.5)
#     """
#     chromato, time_rn = chromato_obj
#     coeffs2 = pywt.dwt2(chromato, 'bior1.3')
#     LL, (LH, HL, HH) = coeffs2
#     titles = ['Approximation', ' Horizontal detail',
#             'Vertical detail', 'Diagonal detail']
#     fig = plt.figure(figsize=(12, 3))
#     for i, a in enumerate([LL, LH, HL, HH]):
#         ax = fig.add_subplot(1, 4, i + 1)
#         ax.contourf(np.transpose(a))
#         ax.set_title(titles[i], fontsize=10)
#     fig.tight_layout()
#     plt.show()
#     return peak_local_max((HH, time_rn), mod_time, seuil)


def clustering(coordinates_all_mass, chromato):
    """
    Applies DBSCAN clustering algorithm to group points based on their spatial
    proximity in the chromatogram.
    The function identifies clusters and returns the coordinates of the points
    with the highest intensity within each cluster.

    Parameters:
        coordinates_all_mass (numpy.ndarray): An array of coordinates (2D or
        3D) where each point represents a mass/retention time pair or triplet.
        chromato (numpy.ndarray): The chromatogram matrix.

    Returns:
        numpy.ndarray: An array of coordinates corresponding to the points
        with the highest intensity in each cluster.

    Example:
        clustering(coordinates_all_mass, chromato)
    """

    if (not len(coordinates_all_mass)):
        return np.array([])
    clustering = DBSCAN(eps=3, min_samples=1).fit(coordinates_all_mass[:, :2])
    # Regroup points per clusters
    clusters = []
    for i in range((np.max(clustering.labels_) + 1)):
        clusters.append([])

    # If points are with radius (blobs)
    if (coordinates_all_mass[0].shape[0] == 3):
        for i, (t1, t2, r) in enumerate(coordinates_all_mass):
            clusters[clustering.labels_[i]].append([t1, t2, r])
    else:
        for i, (t1, t2) in enumerate(coordinates_all_mass):
            clusters[clustering.labels_[i]].append([t1, t2])

    clusters = np.array(clusters)
    # Coordinates of the point with the biggest intensity in the cluster for every clusters
    coordinates = []
    for cluster in clusters:
        if (len(cluster) > 1):
            coord = cluster[np.argmax(np.array([chromato[coord[0], coord[1]] for coord in cluster]))]
        else:
            coord = cluster[0]
        coordinates.append(coord)
    coordinates = np.array(coordinates)
    return np.array(coordinates)

def blob_log_kernel(i, m_chromato, min_sigma, max_sigma, seuil, threshold_abs, num_sigma):
    #blobs_log = blob_log(m_chromato, min_sigma = min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold=threshold_abs)
    blobs_log = skimage.feature.blob_log(m_chromato, min_sigma = min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_abs)

    blobs_log[:, 2] = blobs_log[:, 2] *  math.sqrt(2)
    blobs_log = blobs_log.astype(int)
    return blobs_log

def LoG_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=10, min_sigma=10, max_sigma=30, threshold_abs=0):
    cpu_count = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes = cpu_count)
    coordinates_all_mass = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(blob_log_kernel, [(i, chromato_cube[i], min_sigma, max_sigma, seuil, threshold_abs, num_sigma) for i in range(len(chromato_cube))])):
            for coord in result:
                t1, t2, r = coord
                is_in = False
                for j in range(len(coordinates_all_mass)):
                    m, cam_t1, cam_t2, cam_r = coordinates_all_mass[j]
                    if ([t1, t2] == [cam_t1, cam_t2]):
                        # Keep the element with the biggest radius
                        if (r > cam_r):
                            coordinates_all_mass[j][3] = r
                        is_in = True
                        break
                if (not is_in):
                    coordinates_all_mass.append([i, t1, t2, r])

    return np.array(coordinates_all_mass)

# def LoG_mass_per_mass(chromato_cube, seuil, num_sigma=10, min_sigma=10, max_sigma=30, threshold_abs=0):
#     coordinates_all_mass = []
#     for i in range(chromato_cube.shape[0]):
#         m_chromato = chromato_cube[i]

#         blobs_log = blob_log(m_chromato, min_sigma = min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold=threshold_abs)
#         blobs_log[:, 2] = blobs_log[:, 2] *  math.sqrt(2)

#         blobs_log = blobs_log.astype(int)
#         for coord in blobs_log:
#             t1, t2, r = coord
#             is_in = False
#             for i in range(len(coordinates_all_mass)):
#                 m, cam_t1, cam_t2, cam_r = coordinates_all_mass[i]
#                 if ([t1, t2] == [cam_t1, cam_t2]):
#                     # Keep the element with the biggest radius
#                     if (r > cam_r):
#                         coordinates_all_mass[i][3] = r
#                     is_in = True
#                     break
#             if (not is_in):
#                 coordinates_all_mass.append([i, t1, t2, r])

#     return np.array(coordinates_all_mass)

def LoG(chromato_obj, mod_time, seuil, num_sigma=10, threshold_abs=0, mode="tic", chromato_cube=None, cluster=False, min_sigma=1, max_sigma=30, unique=True):
    """
    Detects blobs in a chromatogram using the Laplacian of Gaussian (LoG) method.
    The function compute their radius, using a multi-scale approach to capture structures at different resolutions.
    
    Parameters:
    ------------
    chromato_obj : tuple
        A tuple containing the chromatographic matrix and the corresponding time values.
    mod_time : float
        Modulation time used for the analysis.
    seuil : float
        Relative threshold used for blob detection.
    num_sigma : int, optional (default=10)
        Number of scales used for blob detection.
    threshold_abs : float, optional (default=0)
        Absolute threshold for blob detection.
    mode : str, optional (default="tic")
        Analysis mode. Can be "tic" (total ion current) or "mass_per_mass".
        - If "mass_per_mass" is selected, blobs are detected separately for each mass spectrum in `chromato_cube` using multiprocessing for efficiency.
    chromato_cube : ndarray, optional (default=None)
        Chromatographic cube containing individual masses if "mass_per_mass" mode is used.
    cluster : bool, optional (default=False)
        Indicates whether clustering should be applied to the detected blobs to group nearby detections.
    min_sigma : int, optional (default=1)
        Minimum sigma value used for blob detection.
    max_sigma : int, optional (default=30)
        Maximum sigma value used for blob detection.
    unique : bool, optional (default=True)
        Indicates whether duplicate blobs should be removed.
    
    Returns: 
    ---------
    tuple (ndarray, ndarray)
        - An array containing the coordinates of the detected blobs (without the radius).
        - An array containing the radius values of the detected blobs.
          * In "tic" mode, these radii correspond to detected blobs in the total ion current.
          * In "mass_per_mass" mode, the radii correspond to blobs detected separately for each mass spectrum in `chromato_cube`.
        - If "mass_per_mass" mode is used, multiprocessing is applied to improve performance when processing multiple mass spectra simultaneously.
    
    Notes:
    ------
        - In "3D" mode, the function processes a 3D chromatographic cube, with the blobs being detected across the entire 3D data set.
        - In "mass_per_mass" mode, the function processes each individual mass spectrum separately in the chromatographic cube, which is useful for analyzing mass spectral data over time.
        - If `cluster=True`, the function applies clustering to group nearby blobs together, which can be useful for merging multiple detections of the same feature.
    """
    
    chromato, time_rn = chromato_obj
    if (mode == "3D"):
        #blobs_log = blob_log(chromato_cube, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold=.1)
        blobs_log = skimage.feature.blob_log(chromato_cube, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_abs)

        #delete mass dimension ([[2 720 128], [24 720 128]] -> [[720 128], [720 128]])
        blobs_log = np.delete(blobs_log, 0, -1)
        blobs_log[:, 2] = blobs_log[:, 2] *  math.sqrt(2)
        blobs_log = blobs_log.astype(int)
        if (unique):
            blobs_log = np.unique(blobs_log, axis=0)
        
        if (cluster == True):
            blobs_log = clustering(blobs_log, chromato)
        return np.delete(blobs_log, 2 ,-1), blobs_log[:,2]

    if (mode == "mass_per_mass"):
        # Coordinates of all mass peaks with their radius
        '''coordinates_all_mass = []
        for i in range(chromato_cube.shape[0]):
            m_chromato = chromato_cube[i]

            blobs_log = blob_log(m_chromato, min_sigma = 10, max_sigma=30, num_sigma=num_sigma, threshold_rel=seuil, threshold=threshold_abs * np.max(m_chromato))
            blobs_log[:, 2] = blobs_log[:, 2] *  math.sqrt(2)

            blobs_log = blobs_log.astype(int)
            for coord in blobs_log:
                is_in = False
                for i in range(len(coordinates_all_mass)):
                    tmp = coordinates_all_mass[i]
                    if ([coord[0], coord[1]] == [tmp[0], tmp[1]]):
                        # Keep the element with the biggest radius
                        if (coord[2] > coordinates_all_mass[i][2]):
                            coordinates_all_mass[i][2] = coord[2]
                        is_in = True
                        break
                if (not is_in):
                    coordinates_all_mass.append(coord)
                #coordinates_all_mass.append(coord)

        coordinates_all_mass = np.array(coordinates_all_mass)'''
        #coordinates_all_mass = LoG_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=num_sigma, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs * np.max(chromato))
        coordinates_all_mass = LoG_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=num_sigma, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs)
        if (len(coordinates_all_mass) == 0):
            return np.array([]),  np.array([])
        coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
        if (unique):
            coordinates_all_mass = np.unique(coordinates_all_mass, axis=0)
                
        if (cluster == True):
            coordinates_all_mass = clustering(coordinates_all_mass, chromato)
        return np.delete(coordinates_all_mass, 2 ,-1), coordinates_all_mass[:,2]
    else:
        max_peak_val = np.max(chromato)
        #blobs_log = blob_log(chromato, min_sigma = 10, max_sigma=30, num_sigma=num_sigma, threshold_rel=seuil, threshold=.1)
        blobs_log = skimage.feature.blob_log(chromato, min_sigma = min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_abs)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] *  math.sqrt(2)
        blobs_log = blobs_log.astype(int)
        blobs_log = np.array(blobs_log)
        return np.delete(blobs_log, 2 ,-1), blobs_log[:,2] 
        #return np.array([[x,y] for x,y,r in blobs_log if chromato[x,y] > seuil * max_peak_val]), np.array([r for x,y,r in blobs_log if chromato[x,y] > seuil * max_peak_val])


def blob_dog_kernel(i, m_chromato, min_sigma, max_sigma, seuil, threshold_abs, sigma_ratio):
    #blobs_dog = blob_dog(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, threshold_rel=seuil, threshold=threshold_abs, sigma_ratio=sigma_ratio)
    blobs_dog = skimage.feature.blob_dog(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, threshold_rel=threshold_abs, sigma_ratio=sigma_ratio)

    blobs_dog[:, 2] = blobs_dog[:, 2] *  math.sqrt(2)
    blobs_dog = blobs_dog.astype(int)
    return blobs_dog

def DoG_mass_per_mass_multiprocessing(chromato_cube, seuil, sigma_ratio=1.6, min_sigma=1, max_sigma=30, threshold_abs=0):
    
    cpu_count = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes = cpu_count)
    coordinates_all_mass = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(blob_dog_kernel, [(i, chromato_cube[i], min_sigma, max_sigma, seuil, threshold_abs, sigma_ratio) for i in range(len(chromato_cube))])):
            for coord in result:
                t1, t2, r = coord
                # Check if there is already this coord and keep the one with the biggest radius
                is_in = False
                for j in range(len(coordinates_all_mass)):
                    m, cam_t1, cam_t2, cam_r = coordinates_all_mass[j]
                    if ([t1, t2] == [cam_t1, cam_t2]):
                        # Keep the element with the biggest radius
                        if (r > cam_r):
                            coordinates_all_mass[j][3] = r
                        is_in = True
                        break
                if (not is_in):
                    coordinates_all_mass.append([i, t1, t2, r])
    return np.array(coordinates_all_mass)

# def DoG_mass_per_mass(chromato_cube, seuil, sigma_ratio=1.6, min_sigma=1, max_sigma=30, threshold_abs=0):
#     coordinates_all_mass = []
#     for i in range(chromato_cube.shape[0]):
#         m_chromato = chromato_cube[i]

#         blobs_dog = blob_dog(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, threshold_rel=seuil, threshold= threshold_abs, sigma_ratio=sigma_ratio)
#         blobs_dog[:, 2] = blobs_dog[:, 2] *  math.sqrt(2)

#         blobs_dog = blobs_dog.astype(int)
#         for coord in blobs_dog:
#             t1, t2, r = coord
#             # Check if there is already this coord and keep the one with the biggest radius
#             is_in = False
#             for i in range(len(coordinates_all_mass)):
#                 m, cam_t1, cam_t2, cam_r = coordinates_all_mass[i]
#                 if ([t1, t2] == [cam_t1, cam_t2]):
#                     # Keep the element with the biggest radius
#                     if (r > cam_r):
#                         coordinates_all_mass[i][3] = r
#                     is_in = True
#                     break
#             if (not is_in):
#                 coordinates_all_mass.append([i, t1, t2, r])
                    
#     return np.array(coordinates_all_mass)

def DoG(chromato_obj, mod_time, seuil, sigma_ratio=1.6, threshold_abs=0, mode="tic", chromato_cube=None, cluster=False, min_sigma=1, max_sigma=30, unique=True):
    """
    Detects blobs in a 2D or 3D chromatogram using the Difference of Gaussians (DoG) method.
    
    Parameters:
    ------------
    chromato_obj : tuple
        A tuple containing the chromatographic matrix and the corresponding time values.
    mod_time : float
        Modulation time used for the analysis.
    seuil : float
        Relative threshold used for blob detection.
    sigma_ratio : float, optional (default=1.6)
        The ratio between the standard deviations of the two Gaussian functions used in the DoG filter.
    threshold_abs : float, optional (default=0)
        Absolute threshold for blob detection.
    mode : str, optional (default="tic")
        Analysis mode. Can be:
        - "tic" (total ion current): Process the entire chromatogram for blob detection.
        - "mass_per_mass": Process each mass spectrum separately.
        - "3D": Process a 3D chromatographic cube for blob detection.
    chromato_cube : ndarray, optional (default=None)
        Chromatographic cube containing individual masses if "mass_per_mass" or "3D" mode is used.
    cluster : bool, optional (default=False)
        Indicates whether clustering should be applied to the detected blobs to group nearby detections.
    min_sigma : int, optional (default=1)
        Minimum sigma value used for blob detection.
    max_sigma : int, optional (default=30)
        Maximum sigma value used for blob detection.
    unique : bool, optional (default=True)
        Indicates whether duplicate blobs should be removed.
    
    Returns: 
    ---------
    tuple (ndarray, ndarray)
        - An array containing the coordinates of the detected blobs (without the radius).
        - An array containing the radius values of the detected blobs.
          * In "tic" mode, these radii correspond to blobs detected in the total ion current.
          * In "mass_per_mass" mode, the radii correspond to blobs detected separately for each mass spectrum in `chromato_cube`.
          * In "3D" mode, blobs are detected in a 3D chromatographic cube and the radii are adjusted accordingly.
        - If "mass_per_mass" or "3D" mode is used, multiprocessing is applied to improve performance when processing multiple mass spectra or cube slices simultaneously.
    """

    chromato, time_rn = chromato_obj
    # Compute DoG on the entire chromato cube
    if (mode == "3D"):
        #delete mass dimension ([[2 720 128], [24 720 128]] -> [[720 128], [720 128]])
        #blobs_dog = blob_dog(chromato_cube, min_sigma=min_sigma, max_sigma=max_sigma, threshold_rel=seuil, threshold=.1, sigma_ratio=sigma_ratio)
        blobs_dog = skimage.feature.blob_dog(chromato_cube, min_sigma=min_sigma, max_sigma=max_sigma, threshold_rel=threshold_abs, sigma_ratio=sigma_ratio)
        
        blobs_dog = np.delete(blobs_dog, 0, -1)
        blobs_dog[:, 2] = blobs_dog[:, 2] *  math.sqrt(2)
        blobs_dog = blobs_dog.astype(int)
        if (unique):
            blobs_dog = np.unique(blobs_dog, axis=0)
        if (cluster == True):
            blobs_dog = clustering(blobs_dog, chromato)
        return np.delete(blobs_dog, 2 ,-1), blobs_dog[:,2]
    if (mode == "mass_per_mass"):
        # Coordinates of all mass peaks with their radius

        #coordinates_all_mass = DoG_mass_per_mass_multiprocessing(chromato_cube, seuil, sigma_ratio=sigma_ratio, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs * np.max(chromato))
        coordinates_all_mass = DoG_mass_per_mass_multiprocessing(chromato_cube, seuil, sigma_ratio=sigma_ratio, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs)

        if (len(coordinates_all_mass) == 0):
            return np.array([]),  np.array([])
        #coordinates_all_mass = DoG_mass_per_mass(chromato_cube, seuil, sigma_ratio=sigma_ratio, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs * np.max(chromato))
        coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
        if (unique):
            coordinates_all_mass = np.unique(coordinates_all_mass, axis=0)
    

        if (cluster == True):
            coordinates_all_mass = clustering(coordinates_all_mass, chromato)
        return np.delete(coordinates_all_mass, 2 ,-1), coordinates_all_mass[:,2]

    else:
        max_peak_val = np.max(chromato)
        #blobs_dog = blob_dog(chromato, min_sigma = 10, max_sigma=30, threshold_rel=seuil,threshold=.1, sigma_ratio=sigma_ratio)
        
        '''PENSER A MODIFIER MIN SIGMA ET MAX SIGMA'''
        
        blobs_dog = skimage.feature.blob_dog(chromato, min_sigma = min_sigma, max_sigma=max_sigma, threshold_rel=threshold_abs, sigma_ratio=sigma_ratio)

        # Compute radii in the 3rd column.
        blobs_dog[:, 2] = blobs_dog[:, 2] *  math.sqrt(2)
        blobs_dog = blobs_dog.astype(int)
        blobs_dog = np.array(blobs_dog)
        return np.delete(blobs_dog, 2 ,-1), blobs_dog[:,2] 
        #return np.array([[x,y] for x,y,r in blobs_dog if chromato[x,y] > seuil * max_peak_val]), np.array([r for x,y,r in blobs_dog if chromato[x,y] > seuil * max_peak_val])

def blob_doh_kernel(i, m_chromato, min_sigma, max_sigma, seuil, threshold_abs, num_sigma):

    #blobs_doh = blob_doh(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold = threshold_abs)
    blobs_doh = blob_doh(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_abs)

    blobs_doh = blobs_doh.astype(int)
    return blobs_doh

def DoH_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=10, min_sigma=10, max_sigma=30, threshold_abs=0):
    cpu_count = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes = cpu_count)
    coordinates_all_mass = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(blob_doh_kernel, [(i, chromato_cube[i], min_sigma, max_sigma, seuil, threshold_abs, num_sigma) for i in range(len(chromato_cube))])):
            for coord in result:
                t1, t2, r = coord
                is_in = False
                for j in range(len(coordinates_all_mass)):
                    m, cam_t1, cam_t2, cam_r = coordinates_all_mass[j]
                    if ([t1, t2] == [cam_t1, cam_t2]):
                        # Keep the element with the biggest radius
                        if (r > cam_r):
                            coordinates_all_mass[j][3] = r
                        is_in = True
                        break
                if (not is_in):
                    coordinates_all_mass.append([i, t1, t2, r])
    return np.array(coordinates_all_mass)

# def DoH_mass_per_mass(chromato_cube, seuil, num_sigma=10, min_sigma=10, max_sigma=30, threshold_abs=0):
#     coordinates_all_mass = []
#     for i in range(chromato_cube.shape[0]):
#         m_chromato = chromato_cube[i]

#         blobs_doh = blob_doh(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold = threshold_abs)
#         blobs_doh = blobs_doh.astype(int)
#         for coord in blobs_doh:
#             t1, t2, r = coord
#             is_in = False
#             for i in range(len(coordinates_all_mass)):
#                 m, cam_t1, cam_t2, cam_r = coordinates_all_mass[i]
#                 if ([t1, t2] == [cam_t1, cam_t2]):
#                     # Keep the element with the biggest radius
#                     if (r > cam_r):
#                         coordinates_all_mass[i][3] = r
#                     is_in = True
#                     break
#             if (not is_in):
#                 coordinates_all_mass.append([i, t1, t2, r])

#     return np.array(coordinates_all_mass)

def DoH(chromato_obj, mod_time, seuil, num_sigma=10, threshold_abs=0, mode="tic", chromato_cube=None, cluster=False, min_sigma=10, max_sigma=30, unique=True):
    """
    Detects blobs in a 2D chromatogram using the Determinant of Hessian (DoH) method.
    
    Parameters:
    ------------
    chromato_obj : tuple
        A tuple containing the chromatographic matrix and the corresponding time values.
    mod_time : float
        Modulation time used for the analysis.
    seuil : float
        Relative threshold used for blob detection.
    num_sigma : int, optional (default=10)
        Number of scales used for blob detection.
    threshold_abs : float, optional (default=0)
        Absolute threshold for blob detection.
    mode : str, optional (default="tic")
        Analysis mode. Can be "tic" (total ion current) or "mass_per_mass".
    chromato_cube : ndarray, optional (default=None)
        Chromatographic cube containing individual masses if "mass_per_mass" mode is used.
    cluster : bool, optional (default=False)
        Indicates whether clustering should be applied to the detected blobs.
    min_sigma : int, optional (default=10)
        Minimum sigma value used for blob detection.
    max_sigma : int, optional (default=30)
        Maximum sigma value used for blob detection.
    unique : bool, optional (default=True)
        Indicates whether duplicate blobs should be removed.
    
    Returns:
    ---------
    tuple (ndarray, ndarray)
        - An array containing the coordinates of the detected blobs (without the radius).
        - An array containing the radius values of the detected blobs.
        Uses multiprocessing for "mass_per_mass" mode to improve performance when processing multiple mass spectra.
    """

    chromato, time_rn = chromato_obj

    if (mode == "mass_per_mass"):
        # Coordinates of all mass peaks with their radius
        '''coordinates_all_mass = []
        for i in range(chromato_cube.shape[0]):
            m_chromato = chromato_cube[i]

            blobs_doh = blob_doh(m_chromato, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=seuil, threshold = threshold_abs * np.max(m_chromato))
            blobs_doh = blobs_doh.astype(int)
            for coord in blobs_doh:
                is_in = False
                for i in range(len(coordinates_all_mass)):
                    tmp = coordinates_all_mass[i]
                    if ([coord[0], coord[1]] == [tmp[0], tmp[1]]):
                        # Keep the element with the biggest radius
                        if (coord[2] > coordinates_all_mass[i][2]):
                            coordinates_all_mass[i][2] = coord[2]
                        is_in = True
                        break
                if (not is_in):
                    coordinates_all_mass.append(coord)
                #coordinates_all_mass.append(coord)

        coordinates_all_mass = np.array(coordinates_all_mass)'''

        #coordinates_all_mass = DoH_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=num_sigma, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs * np.max(chromato))
        coordinates_all_mass = DoH_mass_per_mass_multiprocessing(chromato_cube, seuil, num_sigma=num_sigma, min_sigma=min_sigma, max_sigma=max_sigma, threshold_abs=threshold_abs)

        if (len(coordinates_all_mass) == 0):
            return np.array([]),  np.array([])

        coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
        if (unique):
            coordinates_all_mass = np.unique(coordinates_all_mass, axis=0)
        if (cluster == True):
            coordinates_all_mass = clustering(coordinates_all_mass, chromato)

        return np.delete(coordinates_all_mass, 2 ,-1), coordinates_all_mass[:,2]

    else:
        #blobs_doh = blob_doh(chromato, min_sigma = 10, max_sigma=30, num_sigma=num_sigma, threshold_rel=seuil, threshold=.01)
        blobs_doh = skimage.feature.blob_doh(chromato, min_sigma = 10, max_sigma=30, num_sigma=num_sigma, threshold_rel=threshold_abs)

        blobs_doh = blobs_doh.astype(int)
        blobs_doh = np.array(blobs_doh)
        return np.delete(blobs_doh, 2 ,-1), blobs_doh[:,2]

def pers_hom_kernel(i, m_chromato, seuil, threshold_abs):
    g0 = persistence(m_chromato)
    pts = []
    max_peak_val = np.max(m_chromato)
    for i, homclass in enumerate(g0):
        p_birth, bl, pers, p_death = homclass
        x,y = p_birth
        if (m_chromato[x,y] < threshold_abs * max_peak_val):
            continue
        pts.append((x,y))
    return pts

def pers_hom_mass_per_mass_multiprocessing(chromato_cube, seuil, threshold_abs=0):
    cpu_count = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes = cpu_count)
    coordinates_all_mass = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(pers_hom_kernel, [(i, chromato_cube[i], seuil, threshold_abs) for i in range(len(chromato_cube))])):
            for x,y in result:
                coordinates_all_mass.append([i,x,y])

    return np.array(coordinates_all_mass)

def pers_hom(chromato_obj, mod_time, seuil, threshold_abs=None, mode="tic", cluster=False, chromato_cube=None, unique=True):
    """
    This function applies persistent homology to detect significant peaks in 
    a chromatographic dataset. It supports two modes:
    - "mass_per_mass": Applies persistent homology to each mass slice individually.
    - `tic` (default): Computes persistent homology directly on the chromatogram.

    Parameters:
    -----------
    chromato_obj : tuple (np.ndarray, np.ndarray)
        Tuple containing the chromatogram (2D array) and retention times.
    mod_time : float
        Modulation time for chromatographic analysis.
    seuil : float
        Relative intensity threshold for peak detection.
    threshold_abs : float, optional (default=None)
        Absolute intensity threshold for peak detection.
    mode : str, optional (default="tic")
        Peak detection mode, can be `"mass_per_mass"` or `"tic"`.
    cluster : bool, optional (default=False)
        If `True`, applies clustering to detected peaks.
    chromato_cube : np.ndarray, optional (default=None)
        3D array representing the chromatogram cube (mass, retention time, intensity).
        Required for `"mass_per_mass"` mode.
    unique : bool, optional (default=True)
        If `True`, ensures detected peaks are unique.

    Returns:
    --------
    np.ndarray
        - If mode is `"mass_per_mass"`: Returns detected peak coordinates across mass slices, using multiprocessing.
        - If mode is `"tic"`: Returns significant peak coordinates based on persistent homology.
    """

    chromato, time_rn = chromato_obj
    if (mode == "mass_per_mass"):
        #chromato_cube = chromato_cube[:10]
        coordinates_all_mass = pers_hom_mass_per_mass_multiprocessing(chromato_cube, seuil, threshold_abs=threshold_abs)
        # We delete masse dimension
        if (len(coordinates_all_mass) > 0):
            coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
            if (unique):
                coordinates_all_mass = np.unique(coordinates_all_mass, axis=0)
        if (cluster == False):
            return coordinates_all_mass
        return clustering(coordinates_all_mass, chromato)
    else:
        g0 = persistence(chromato)
        pts = []
        max_peak_val = np.max(chromato)
        for i, homclass in enumerate(g0):
            p_birth, bl, pers, p_death = homclass
            x,y = p_birth
            '''if (chromato[x,y] < seuil * max_peak_val):
                continue'''
            pts.append((x,y))
        return np.array(pts)

def plm_kernel(i, m_chromato, min_distance, seuil, threshold_abs):
    #return peak_local_max(m_chromato, min_distance=min_distance, threshold_rel=seuil, threshold_abs=threshold_abs)
    return skimage.feature.peak_local_max(m_chromato, min_distance=min_distance, threshold_rel=threshold_abs)


def plm_mass_per_mass_multiprocessing(chromato_cube, seuil, min_distance=1, threshold_abs=0):
    """
    Detects peaks in each mass slice of a chromatogram cube using multiprocessing.

    This function applies the `plm_kernel()` function in parallel to each mass
    slice of the chromatogram cube, distributing the workload across multiple CPU cores.

    Parameters:
    -----------
    chromato_cube : np.ndarray
        3D array representing the chromatogram cube (mass, retention time, intensity).
    seuil : float
        Relative intensity threshold for peak detection.
    min_distance : int, optional (default=1)
        Minimum distance between detected peaks.
    threshold_abs : float, optional (default=0)
        Absolute intensity threshold for peak detection.

    Returns:
    --------
    np.ndarray
        An array of detected peak coordinates, each row containing:
        [mass index, retention time, intensity].
    """

    cpu_count = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes = cpu_count)
    coordinates_all_mass = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for i, result in enumerate(pool.starmap(plm_kernel, [(i, chromato_cube[i], min_distance, seuil, threshold_abs) for i in range(len(chromato_cube))])):
            for x,y in result:
                coordinates_all_mass.append([i,x,y])

    return np.array(coordinates_all_mass)


# # Return 3D coordinates so we can filter with relative threshold after without recomputing
# # Here threshold_abs isn't a ratio but the flat value
# def plm_mass_per_mass(chromato_cube, seuil, min_distance=1, threshold_abs=0):

#     coordinates_all_mass = []
#     for i in range(chromato_cube.shape[0]):
#         m_chromato = chromato_cube[i]
#         coordinates = peak_local_max(m_chromato, min_distance=min_distance, threshold_rel=threshold_abs)
#         #coordinates = peak_local_max(m_chromato, min_distance=min_distance, threshold_rel=seuil, threshold_abs=threshold_abs)

#         for x,y in coordinates:
#             coordinates_all_mass.append([i,x,y])

#     return np.array(coordinates_all_mass)

def peak_local_max(chromato_obj, mod_time, dynamic_threshold_fact,  min_distance=1, mode="tic", chromato_cube=None, cluster=False, threshold_abs=0, unique=True):
    """
    #ancienne fct nommee plm
    Detects peaks in a chromatographic dataset using different processing modes.

    This function identifies local maxima (peaks) in chromatographic data using 
    `skimage.feature.peak_local_max()` with multiple modes:
    - `"3D"`: Analyzes the entire chromatogram cube.
    - `"mass_per_mass"`: Detects peaks slice by slice across mass dimensions (parallelized).
    - `"tic"`: Extracts peaks using Total Ion Chromatogram (TIC) mode.

    Parameters:
    -----------
    chromato_obj : tuple (np.ndarray, np.ndarray)
        Tuple containing the chromatogram (2D array) and retention times.
    mod_time : float
        Modulation time for chromatographic analysis.
    dynamic_threshold_fact : float
        Relative intensity threshold for peak detection.
    min_distance : int, optional (default=1)
        Minimum distance between detected peaks.
    mode : str, optional (default="tic")
        Peak detection mode, can be `"3D"`, `"mass_per_mass"`, or `"tic"`.
    chromato_cube : np.ndarray, optional (default=None)
        3D array representing the chromatogram cube (mass, retention time, intensity).
        Required for `"3D"` and `"mass_per_mass"` modes.
    cluster : bool, optional (default=False)
        If `True`, applies clustering to detected peaks.
    threshold_abs : float, optional (default=0)
        Absolute intensity threshold for peak detection.
    unique : bool, optional (default=True)
        If `True`, ensures detected peaks are unique.

    Returns:
    --------
    np.ndarray
        Array of detected peak coordinates. The shape depends on the selected mode:
        - In `"3D"` mode: [[retention time, intensity]]
        - In `"mass_per_mass"` mode: [[mass index, retention time, intensity]]
        - In `"tic"` mode: [[retention time, intensity]]

    Notes:
        If both threshold_abs and threshold_rel are provided, the maximum of the two is chosen as the minimum intensity threshold of peaks.
    """
    chromato, time_rn = chromato_obj

    '''if (threshold_abs != None):
        threshold_abs = np.max(chromato) * threshold_abs'''

    # Compute peak_local_max on the entire chromato cube (3D peak_local_max)
    if (mode == "3D"):
        #cube_coordinates = peak_local_max(chromato_cube, threshold_rel=seuil)
        cube_coordinates = skimage.feature.peak_local_max(chromato_cube, threshold_rel=threshold_abs)
        
        #delete mass dimension ([[2 720 128], [24 720 128]] -> [[720 128], [720 128]])
        coordinates = np.delete(cube_coordinates, 0, -1)
        if (unique):
            #merge same points
            coordinates = np.unique(coordinates, axis=0)
        if (cluster == False):
            return coordinates
        return clustering(coordinates, chromato)

    # Compute peak_local_max for very mass slices in the chromato cube 
    elif (mode == "mass_per_mass"):
       
        coordinates_all_mass = plm_mass_per_mass_multiprocessing(chromato_cube, dynamic_threshold_fact, min_distance=1, threshold_abs=threshold_abs)
        # We delete masse dimension
        if (len(coordinates_all_mass) > 0):
            coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
            if (unique):
                coordinates_all_mass = np.unique(coordinates_all_mass, axis=0)
        if (cluster == False):
            return coordinates_all_mass
        return clustering(coordinates_all_mass, chromato)
    # Use TIC
    else:
        coordinates = skimage.feature.peak_local_max(chromato, min_distance=min_distance, threshold_rel=dynamic_threshold_fact)
        return coordinates


def peak_detection(chromato_obj, spectra, chromato_cube, dynamic_threshold_fact, 
                   ABS_THRESHOLDS,
                   mod_time, method="persistent_homology", mode='tic',
                   cluster=True, min_distance=1, sigma_ratio=1.6, num_sigma=10,
                   unique=True):
    r"""Detect peaks in a 2D or 3D chromatogram.

    Parameters
    ----------
    chromato_obj:
        chromato, time_rn, spectra_obj = chromato_obj
    chromato_cube: ndarray
        3D Chromatogram.
    seuil: float
        Threshold to filter peaks. A float between 0 and 1. A peak is returned
        if its intensity in chromato is greater than the maximum value in the 
        chromatogram multiply by seuil.
    ABS_THRESHOLDS: optional
        If mode='mass_per_mass' or mode='3D', ABS_THRESHOLDS is the threshold
        relative to a slice of the 3D chromatogram or a slice of the 3D
        chromatogram.
    mod_time: optional
        The modulation time of the chromatogram.
    method: optional
        The method to use. The default method is "persistent_homology" but it
        can be "peak_local_max", "LoG", "DoG", or "DoH".
    cluster: optional
        Whether to cluster or not the 3D coordinates when mode='mass_per_mass'
        or mode='3D'. When peaks are detected in each slice of the 3D
        chromatogram, coordinates associated to the same peak may differ a
        little. 3D peak coordinates (rt1, rt2, mass slice) which are really
        close in the first two dimensions are merged and the coordinates with
        the highest intensity in the TIC chromatogram is kept.
    min_distance: optional
        peak_local_max method parameter. The minimal allowed distance
        separating peaks. To find the maximum number of peaks, use
        min_distance=1.
    sigma_ratio: optional
        DoG method parameter. The ratio between the standard deviation of
        Gaussian Kernels used for computing the Difference of Gaussians.
    num_sigma: optional
        LoG/DoH method parameter. The number of intermediate values of
        standard deviations to consider between min_sigma (1) and max_sigma
        (30).
    Returns
    -------
    A: ndarray
        Detected peaks. ndarray of coordinates.
    Examples
    --------
    >>> # Detect peak in the TIC chromatogram using persistent homology method
    >>> import peak_detection
    >>> import read_chroma
    >>> from skimage.restoration import estimate_sigma
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> # seuil=MIN_SEUIL is computed as the ratio between 5 times the
    estimated gaussian white noise standard deviation (sigma) in the
    chromatogram and the max value in the chromatogram.
    >>> sigma = estimate_sigma(chromato, channel_axis=None)
    >>> MIN_SEUIL = 5 * sigma * 100 / np.max(chromato)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(
    full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> coordinates = peak_detection.peak_detection(chromato_obj=(chromato,
    time_rn, spectra_obj), chromato_cube=chromato_cube, seuil=MIN_SEUIL)

    """
    chromato, time_rn, spectra_obj = chromato_obj
    max_peak_val = np.max(chromato)
    radius = None
    if (method == "peak_local_max"):
        coordinates = peak_local_max(chromato_obj=(
                    chromato, time_rn), mod_time=mod_time,
                    dynamic_threshold_factor=dynamic_threshold_fact,
                    min_distance=min_distance, mode=mode,
                    chromato_cube=chromato_cube, cluster=cluster,
                    threshold_abs=ABS_THRESHOLDS, unique=unique)
    elif (method == "DoG"):
        coordinates, radius = DoG(chromato_obj=(
                    chromato, time_rn), mod_time=mod_time,
                    dynamic_threshold_factor=dynamic_threshold_fact,
                    sigma_ratio=sigma_ratio, mode=mode,
                    chromato_cube=chromato_cube, cluster=cluster,
                    threshold_abs=ABS_THRESHOLDS, unique=unique)
    elif (method == "LoG"):
        coordinates, radius = LoG(chromato_obj=(
                    chromato, time_rn), mod_time=mod_time,
                    dynamic_threshold_factor=dynamic_threshold_fact,
                    num_sigma=num_sigma, mode=mode,
                    chromato_cube=chromato_cube,
                    cluster=cluster, threshold_abs=ABS_THRESHOLDS,
                    unique=unique)
    elif (method == "DoH"):
        coordinates, radius = DoH(chromato_obj=(
                    chromato, time_rn), mod_time=mod_time,
                    dynamic_threshold_factor=dynamic_threshold_fact,
                    num_sigma=num_sigma, mode=mode,
                    chromato_cube=chromato_cube, cluster=cluster,
                    threshold_abs=ABS_THRESHOLDS, unique=unique)
    elif (method == "persistent_homology"):
        coordinates = pers_hom(chromato_obj=(
                chromato, time_rn), mod_time=mod_time,
                dynamic_threshold_fact=dynamic_threshold_fact, mode=mode,
                chromato_cube=chromato_cube, cluster=cluster,
                threshold_abs=ABS_THRESHOLDS, unique=unique)
    else:
        print("Unknown method")
        return None
    coordinates = np.array(
            [[x, y] for x, y in coordinates if chromato[x, y]
             > dynamic_threshold_fact * max_peak_val])
    return coordinates


'''def peak_detection(chromato_obj, mod_time, method = "peak_local_max", seuil = 0):
    if (method == "peak_local_max"):
        return peak_local_max(chromato_obj, mod_time, seuil)
    elif (method == "wavelets"):
        return wavelet(chromato_obj, mod_time, seuil)
    elif (method == "tf"):
        tf(chromato_obj, mod_time, seuil)
        return None
    elif(method == "sobel"):
        return sobel(chromato_obj, mod_time, seuil)
    elif(method == "gauss_multi_deriv"):
        return gauss_multi_deriv(chromato_obj, mod_time, seuil)
    elif(method == "gauss_laplace"):
        return gauss_laplace(chromato_obj, mod_time, seuil)
    elif(method == "prewitt"):
        return prewitt(chromato_obj, mod_time, seuil)
    elif(method=="gaussian_filter"):
        return gaussian_filter(chromato_obj, mod_time, seuil)
    elif(method=="LoG"):
        return LoG(chromato_obj, mod_time, seuil)
    elif(method=="DoG"):
        return DoG(chromato_obj, mod_time, seuil)
    elif(method=="DoH"):
        return DoH(chromato_obj, mod_time, seuil)
    elif(method=="persistent_homology"):
        return pers_hom(chromato_obj, mod_time, seuil)
    else:
        return None'''