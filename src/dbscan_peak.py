import projection
import numpy as np
import mass_spec
from sklearn.cluster import DBSCAN
from skimage.restoration import estimate_sigma
from sklearn.metrics.pairwise import cosine_distances
import multiprocessing
from multiprocessing import Pool
import skimage
import gc
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from whittaker_eilers import WhittakerSmoother


# parameters to optimmize: 
#clustering
#penaltyrt1 et 2 pour l'laignement intra mass
#DBSCAN threshold intra-mass
#penaltyrt1 et 2 pour l'laignement inter mass
#seuil db scan inter mass

# peak detection
# savitzky golay filter 
# correction ligne de base 
# relative threshold per mass 
# abs seuil (dependant du chorma)
# max number of peaks per mass
# sigma Dog parameter 
 
def detection_mass_par_mass_Dog(chromato_cube,chromato_obj,mod_time,
                                                abs_threshold=1000,
                                                rel_threshold=0.0001,
                                                noise_factor=3,
                                                min_sigma=1,
                                                max_sigma=20,
                                                sigma_ratio=2,
                                                overlap=0.5, 
                                                max_peak_per_mass=600,
                                                rt1_delta=2, 
                                                rt2_delta=0.02,
                                                min_size_cluster_mass=2, 
                                                thr_debscan= 0.02, 
                                                multi_processing=True,
                                                cleaning_close_peak=True):
    r"""
    Detects chromatographic peaks in a 3D data cube (retention time 1 × retention time 2 × m/z)
    on a per-mass (m/z) basis using DoG method.

    Parameters
    ----------
    chromato_cube : np.ndarray
        3D numpy array representing the chromatographic data cube 
        with shape (rt1, rt2, mz), where each voxel contains an intensity value.
        
    chromato_obj : object
        (chromato, time_rn)
        
    mod_time : float
        Modulation time (e.g., for GC GC) in seconds.

    abs_threshold : float, optional
        Absolute intensity threshold for peak detection.
        
    rel_threshold : float, optional
        Relative intensity threshold (e.g., fraction of maximum) for dynamic filtering.

    noise_factor : float, optional
        Multiplier applied to the estimated noise level to define the dynamic threshold.

    min_sigma : int, optional
        parameter of  skimage.feature.blob_dog function

    max_sigma : int, optional
        parameter of  skimage.feature.blob_dog function

    sigma_ratio : float, optional
        parameter of  skimage.feature.blob_dog function

    overlap : float, optional
        parameter of  skimage.feature.blob_dog function

    max_peak_per_mass : int, optional
        Maximum number of peaks retained per m/z slice.

    rt1_delta : float, optional
        Tolerance window in the first retention time dimension for merging or filtering peaks.

    rt2_delta : float, optional
        Tolerance window in the second retention time dimension for merging or filtering peaks.

    min_size_cluster_mass : int, optional
        Minimum number of peaks in a mass cluster (for downstream clustering or filtering).

    thr_debscan : float, optional
        Distance threshold used for density-based clustering (e.g., DBSCAN) during mass alignment.

    multiprocessing : bool, optional
        If True, process m/z slices in parallel to accelerate detection.

    cleaning_close_peak: bool, optional
        If True, start a second clusturing to merge duplicate peaks, and TODO, peaks cut but a modulation (present at RT2 mod_time and 0) 

    Returns
    -------
    list coordinates in index in the chormato
    """ 
    (chromato, time_rn) = chromato_obj

    print("start peak detection")
    results= detect_peak_dog_mp(chromato_cube,abs_threshold,
                       rel_threshold,
                       noise_factor,
                       min_sigma, max_sigma,
                       sigma_ratio,
                       overlap,multi_processing=multi_processing)
    
    print("cluster_per_mass ")
    results, radius_cluster, clusters_cluster, clusters_label_cluster =cluster_per_mass(results,chromato_cube,time_rn,mod_time,rt1_delta=5, rt2_delta=0.1,thr_debscan=0.05,max_peak_per_mass=max_peak_per_mass)
    
    print("deconvolution per mass")

    
    results = [res for res in results if res is not None]
    coordinates_all_mass=[]
    for elt in results:
        for x,y,z in elt:
            coordinates_all_mass.append([x,y,z])
    
    coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
    
    print("compute distance metric "+ str(len(coordinates_all_mass)) + " peaks")
    distance_matrix=compute_distance_metric(coordinates_all_mass,chromato_cube,mod_time,time_rn,rt1_delta=rt1_delta, rt2_delta=rt2_delta)
      
    print("start clustering")
    coordinates, radius, clusters,label= cluster_peak(distance_matrix,chromato,coordinates_all_mass,thr_debscan=thr_debscan,min_sample_db_scan=min_size_cluster_mass)
    print(str(len(coordinates)) + " peaks clustered")
    if(cleaning_close_peak):
        # cluster close peaks 
        distance_matrix=compute_distance_metric(coordinates,chromato_cube,mod_time,time_rn,rt1_delta=2*mod_time, rt2_delta=0.05)
        coordinates, clusters,label= cluster_peak(distance_matrix,chromato,coordinates,thr_debscan=0.02,min_sample_db_scan=1)
        # merge peaks cut but the modulation
        coordinates_in_chromato=projection.matrix_to_chromato(coordinates, time_rn, mod_time, chromato.shape)
        # 1.Identifier les lignes à supprimer (bound = up_bound + low_bound)
        mask = (coordinates_in_chromato[:, 1] > (mod_time-0.05)) | (coordinates_in_chromato[:, 1] < 0.05)
        bound= coordinates[mask]    
        # 2.cluster without RT2 penalty 
        distance_matrix=compute_distance_metric(bound,chromato_cube,mod_time,time_rn,rt1_delta=mod_time, rt2_delta=100000)
        bound_cluster, clusters,label= cluster_peak(distance_matrix,chromato,bound,thr_debscan=0.02,min_sample_db_scan=1)
        # 3. replace cluster 
        coordinates = np.concatenate((coordinates[~mask], bound_cluster), axis=0)
        print(str(len(coordinates))+ " detected peaks after filter")
    return coordinates

# def rt_penalty(rt_vals, rt1_delta=5, rt2_delta=0.1): # rt en seconde !! 
#     rt1 = rt_vals[:, 0][:, None]
#     rt2 = rt_vals[:, 0][None, :]
#     rt1_penalty = np.abs(rt1 - rt2) / rt1_delta # 1 penalité pour 5 secondes

#     rt3 = rt_vals[:, 1][:, None]
#     rt4 = rt_vals[:, 1][None, :]
#     rt2_penalty = np.abs(rt3 - rt4) / rt2_delta #e 1 en RT2 0.1 seconde
#     return rt1_penalty + rt2_penalty

def rt_penalty(rt_vals, rt1_delta=5, rt2_delta=0.1):
    # Normalize RTs by their respective deltas (acts like weighting)
    scaled_rt = rt_vals / np.array([rt1_delta, rt2_delta])
    
    # Compute Manhattan (L1) distance between all rows
    penalty_matrix = cdist(scaled_rt, scaled_rt, metric='cityblock')  # sum of |ΔRT1| + |ΔRT2|
    return penalty_matrix

def compute_distance_metric(coordinates,chromato_cube,mod_time,time_rn,rt1_delta,rt2_delta): # ajouter buffer si trop gros
    intensity_values_list = []
    for i, coordinate in enumerate(coordinates):
        int_values = mass_spec.read_spectrum_from_chromato_cube(coordinate, chromato_cube=chromato_cube)
        rt_values= projection.matrix_to_chromato(coordinates[[i]], time_rn, mod_time, chromato_cube[0,:,:].shape)
        intensity_values_list.append(np.concatenate(([rt_values[0,0]*60,rt_values[0,1]], int_values)))

    intensity_values_list = np.array(intensity_values_list)
    rt_vals = intensity_values_list[:, :2]  # shape (n, 2)
    spectra = intensity_values_list[:, 2:]  # shape (n, m)
    
    rt_penalty_value= rt_penalty(rt_vals,rt1_delta,rt2_delta)
    npeaks = spectra.shape[0]
    if(npeaks <10000):
        spec_dists = cosine_distances(spectra)
        distance_matrix = spec_dists + 0.01 *rt_penalty_value
    else :    
        spec_dists=compute_cosine_distance_batch(spectra)
        spec_dists = spec_dists.astype(np.float32)
        rt_penalty_value = rt_penalty_value.astype(np.float32)
        mask = rt_penalty_value < 10

        # Allocate result only for close RT 
        distance_matrix = np.full_like(spec_dists, 1)
        distance_matrix[mask] = spec_dists[mask] + 0.01 * rt_penalty_value[mask]
    return distance_matrix

import numpy as np
import gc

def compute_cosine_distance_batch(spectra,batch_size=10000):
    n_samples = spectra.shape[0]

    # Optional: use float32 to reduce memory (only if precision is acceptable)
    spectra = spectra.astype(np.float32)

    # Preallocate output matrix
    spec_dists = np.zeros((n_samples, n_samples), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        # Compute distances from batch to all
        dist_batch = cosine_distances(spectra[start:end], spectra)
        spec_dists[start:end, :] = dist_batch

        del dist_batch
        gc.collect()

    return spec_dists


# def cluster_peak(distance_matrix,chromato,coordinates, thr_debscan,min_sample_db_scan):
#     clustering = DBSCAN(eps=thr_debscan, min_samples=min_sample_db_scan, metric='precomputed').fit(distance_matrix)
#     labels = clustering.labels_
#     unique_labels = set(labels)
#     unique_labels.discard(-1)  # Ignore noise
#     ncluster=len(unique_labels)
#     clusters = []
#     clusters_label=[]
#     for i in range(ncluster):
#         clusters.append([])
#         clusters_label.append([])
#     for i, (t1, t2) in enumerate(coordinates):
#         if(clustering.labels_[i]!=-1):
#             clusters[clustering.labels_[i]].append([t1, t2])
#             clusters_label[clustering.labels_[i]].append(i)
#     coordinates = []
#     for cluster in clusters:
#         if (len(cluster) > 1):
#             coord = cluster[np.argmax(np.array([chromato[coord[0], coord[1]] for coord in cluster]))]
#         else:
#             coord = cluster[0]
#         coordinates.append(coord)
#     coordinates = np.array(coordinates)
#     return coordinates, clusters,clusters_label


def cluster_peak(distance_matrix,chromato,coordinates, radius, thr_debscan,min_sample_db_scan):
    clustering = DBSCAN(eps=thr_debscan, min_samples=min_sample_db_scan, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)
    ncluster=len(unique_labels)
    clusters = []
    clusters_label=[]
    for i in range(ncluster):
        clusters.append([])
        clusters_label.append([])
    for i in range(len(coordinates)):
        if(clustering.labels_[i]!=-1):
            clusters[clustering.labels_[i]].append(([coordinates[i],radius[i]]))
            clusters_label[clustering.labels_[i]].append(i)
    coordinates_clus = []
    radius_clus = []
    for cluster in clusters:
        if (len(cluster) > 1):
            coord = cluster[np.argmax(np.array([chromato[coord[0][0], coord[0][1]] for coord in cluster]))][0]
            rad = cluster[np.argmax(np.array([chromato[coord[0][0], coord[0][1]] for coord in cluster]))][1]
        else:
            coord = cluster[0][0]
            rad= cluster[0][1]
        coordinates_clus.append(coord)
        radius_clus.append(rad)
    coordinates_clus = np.array(coordinates_clus)
    radius_clus = np.array(radius_clus)
    return coordinates_clus, radius_clus , clusters, clusters_label


import math 

def detect_peak_dog( chromato_tic,
                    abs_threshold,
                    rel_threshold,
                    noise_factor,
                    min_sigma, max_sigma,
                    sigma_ratio,
                    overlap):

        sigma = estimate_sigma(chromato_tic, channel_axis=None)
        intensity_threshold = intensity_threshold_decision_rule(
            abs_threshold, rel_threshold, noise_factor, sigma, chromato_tic)
        blobs_dog = skimage.feature.blob_dog(chromato_tic, min_sigma=min_sigma,
                                             max_sigma=max_sigma, 
                                             overlap=overlap,
                                             threshold=intensity_threshold,
                                             sigma_ratio=sigma_ratio)
        # Compute radii in the 3rd column.
        #blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)
        
        #blobs_dog = np.array(blobs_dog)
        #coordinates, radius = np.delete(blobs_dog, 2, -1), blobs_dog[:, 2]
        
        # blobs_dog shape: (N, 3), where columns are (y, x, sigma)
        # Adjust radii: radius = sigma * sqrt(2)
        radii = blobs_dog[:, 2] * math.sqrt(2)

        # Keep coordinates as float (y,x)
        blobs_dog = np.rint(blobs_dog).astype(int)
        coordinates = np.array(blobs_dog[:, :2])
        
        index = [j for j,coord in enumerate(coordinates) if coord[0] != 0]
        return coordinates[index] , radii[index]


def detect_peak_dog_mp(chromato_cube,
                       abs_threshold=500,
                       rel_threshold=0.00001,
                       noise_factor=3,
                       min_sigma=1, max_sigma=20,
                       sigma_ratio=2,
                       overlap=0.5,multi_processing=True):

    inputs = range(chromato_cube.shape[0])
    results=[]
    if(multi_processing):
        num_workers = min(multiprocessing.cpu_count(),32)
        with multiprocessing.Pool(processes = num_workers) as pool:
                    for i, result in enumerate(pool.starmap(detect_peak_dog, [(chromato_cube[mass,:,:],
                                                                                abs_threshold,
                                                                                rel_threshold,
                                                                                noise_factor,
                                                                                min_sigma, max_sigma,
                                                                                sigma_ratio,
                                                                                overlap) for mass in inputs])):
                                                                                results.append(result)
    else:
          for mass in inputs:
            tmp= chromato_cube[mass,:,:]
            result=detect_peak_dog(tmp,
                            abs_threshold,
                            rel_threshold,
                            noise_factor,
                            min_sigma, max_sigma,
                            sigma_ratio,
                            overlap)
            results.append(result)                                                                            
    return results


def cluster_per_mass(coordinate,radius,baseline_cube,chromato_cube,time_rn, mod_time,rt1_delta, rt2_delta,thr_debscan,max_peak_per_mass):
    coordinate_cluster=[]
    radius_cluster=[]
    for i in range(len(coordinate)):
        coordinate_cluster.append([])
        radius_cluster.append([])
    for mass in range(len(coordinate)) :
        coord_m=coordinate[mass]
        rad_m= radius[mass]
        tmp=baseline_cube[mass,:,:]
        npeak=len(coord_m)
        if(npeak!=0):
            # peak clusturing
            distance_matrix=compute_distance_metric(coord_m,chromato_cube,mod_time,time_rn,rt1_delta,rt2_delta)
            coord_cluster, rad_cluster , clusters, clusters_label= cluster_peak(distance_matrix,tmp,coord_m,rad_m,thr_debscan, min_sample_db_scan=1)

            if npeak > max_peak_per_mass:
                intensities = np.array([tmp[coord[0], coord[1]] for coord in coord_cluster])
                top_indices = np.argsort(intensities)[-max_peak_per_mass:][::-1]  # descending order
                coord_cluster = coord_cluster[top_indices]
                rad_cluster=rad_cluster[top_indices]
            
            # merge peaks cut but the modulation
            coordinates_in_chromato=projection.matrix_to_chromato(coord_cluster, time_rn,mod_time, tmp.shape)
            # 1.Identifier les lignes à supprimer (bound = up_bound + low_bound)
            mask = (coordinates_in_chromato[:, 1] > (mod_time-0.05)) | (coordinates_in_chromato[:, 1] < 0.05)
            if(any(mask)):
                bound= coord_cluster[mask] 
                radius_bound= rad_cluster[mask]      
                        # 2.cluster without RT2 penalty 
                distance_matrix=compute_distance_metric(bound,chromato_cube,mod_time,time_rn,rt1_delta=mod_time, rt2_delta=100000)
                bound_cluster, radius_bound_cluster, clusters, clusters_label = cluster_peak(distance_matrix,tmp,bound,radius_bound,thr_debscan=thr_debscan,min_sample_db_scan=1)
                        # 3. replace cluster 
                
                coord_cluster = np.concatenate((coord_cluster[~mask], bound_cluster), axis=0)
                rad_cluster = np.concatenate((rad_cluster[~mask], radius_bound_cluster), axis=0)
        else :
            coord_cluster = []
            rad_cluster = []
        coordinate_cluster[mass] = coord_cluster
        radius_cluster[mass]=rad_cluster
    return coordinate_cluster, radius_cluster


# peak_local_max detection
    # intensity_threshold = intensity_threshold_decision_rule(
    #         abs_threshold=1500, rel_threshold=0.0000001, noise_factor=3, sigma=sigma, chromatogram=tmp)
    # coordinates = skimage.feature.peak_local_max(
    #         tmp,
    #         min_distance=1,
    #         threshold_abs=intensity_threshold)
    


# def square(x,y):
#     print(f"Processing {x} in PID {multiprocessing.current_process().pid}")
#     time.sleep(1)  # Simulate work
#     return x * y

# def multi_process_test(x,y):
#     inputs = range(x)  # List of inputs
#     num_workers = 4          # Number of parallel processes
#     start = time.time()
#     res=[]
#     with multiprocessing.Pool(processes = num_workers) as pool:
#         for i, result in enumerate(pool.starmap(square, [(i,y) for i in inputs])):
#             res.append(result)
#     print("Results:", res)
#     print("Total time:", round(time.time() - start, 2), "seconds")
#     return res


        # optimal_lambda_list=[]
        # # find optimal lambda for smoothing 
        # for j in range(0,chromato_tic.shape[1],10):
        #     data_to_smooth = chromato_tic[:,j]
        #     smoother = WhittakerSmoother(lmbda=10, order=1, data_length=len(data_to_smooth))
        #     results = smoother.smooth_optimal(data_to_smooth, break_serial_correlation=True)
        #     optimal_lambda = results.get_optimal().get_lambda()
        #     optimal_lambda_list.append(optimal_lambda)
        
        # optimal_lambda=np.quantile(optimal_lambda_list,0.25)

        # if optimal_lambda >10 :
        #     optimal_lambda=10
        # elif optimal_lambda <5:
        #     optimal_lambda=5

        # # baseline and smooting    
        # correct=baseline_correct(chromato_tic,block_size=10,gamma=0.25, lmbd=optimal_lambda)

        # chromato_tic=correct 

def intensity_threshold_decision_rule(
        abs_threshold,
        rel_threshold,
        noise_factor,
        sigma,
        chromatogram):
    """
    Compute the intensity threshold for peak detection based on the provided
    absolute and relative thresholds, as well as the dynamic noise factor.
    Parameters
    ----------
    abs_threshold : float
        Absolute threshold for peak detection.
    rel_threshold : float
        Relative threshold for peak detection.
    noise_factor : float
        Noise factor for dynamic thresholding.
    chromatogram : ndarray
        Chromatogram data, TIC or 3D.
    Returns
    -------
    float
        The computed intensity threshold for peak detection.
    """
    max_peak_val = np.max(chromatogram)
    dynamic_noise_factor = noise_factor * sigma  
    # if chromatogram is very noisy : avoid detecting noise as if it were real
    # peaks.
    # if chonmatogram  is very clean: detect weaker peaks

    intensity_threshold = max(abs_threshold, rel_threshold * max_peak_val, dynamic_noise_factor)
    return intensity_threshold
