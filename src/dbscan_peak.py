from peak_detection import peak_detection
import projection
import numpy as np
import mass_spec
from sklearn.cluster import DBSCAN
from skimage.restoration import estimate_sigma
from sklearn.metrics.pairwise import cosine_distances
import multiprocessing
from multiprocessing import Pool
import skimage
from peak_detection import intensity_threshold_decision_rule


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
 


def detection_mass_par_mass(chromato_cube,chromato_obj,mod_time,
                                                abs_threshold=500,
                                                rel_threshold=0.00001,
                                                noise_factor=3,
                                                min_sigma=1,
                                                max_sigma=20,
                                                sigma_ratio=2,
                                                overlap=0.5, 
                                                rt1_delta=2, 
                                                rt2_delta=0.02,
                                                min_size_cluster_mass=2, 
                                                thr_debscan= 0.02, 
                                                parallel=True):
    (chromato, time_rn) = chromato_obj

    print("start peak detection")
    results= detect_peak_dog_mp(chromato_cube,abs_threshold,
                       rel_threshold,
                       noise_factor,
                       min_sigma, max_sigma,
                       sigma_ratio,
                       overlap,parallel=parallel)
    
    print("cluster_per_mass")
    results=cluster_per_mass(results,chromato_cube,time_rn)
    results = [res for res in results if res is not None]
    coordinates_all_mass=[]
    for elt in results:
        for x,y,z in elt:
            coordinates_all_mass.append([x,y,z])
    
    coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
    
    print("compute distance metric")
    distance_matrix=compute_distance_metric(coordinates_all_mass,chromato_cube,mod_time,time_rn,rt1_delta=rt1_delta, rt2_delta=rt2_delta)
    
    print("start clustering")
    coordinates, clusters= cluster_peak(distance_matrix,chromato,coordinates_all_mass,thr_debscan=thr_debscan,min_sample_db_scan=min_size_cluster_mass)
    distance_matrix=compute_distance_metric(coordinates,chromato_cube,mod_time,time_rn,rt1_delta=2*1.7, rt2_delta=0.1)
    coordinates, clusters= cluster_peak(distance_matrix,chromato,coordinates_all_mass,thr_debscan=0.03,min_sample_db_scan=1)
    
    print(str(len(coordinates))+ " detected peaks")
    return coordinates

def rt_penalty(rt_vals, rt1_delta=5, rt2_delta=0.1): # rt en seconde !! 
    rt1 = rt_vals[:, 0][:, None]
    rt2 = rt_vals[:, 0][None, :]
    rt1_penalty = np.abs(rt1 - rt2) / rt1_delta # 1 penalité pour 5 secondes

    rt3 = rt_vals[:, 1][:, None]
    rt4 = rt_vals[:, 1][None, :]
    rt2_penalty = np.abs(rt3 - rt4) / rt2_delta #e 1 en RT2 0.1 seconde
    return rt1_penalty + rt2_penalty

def compute_distance_metric(coordinates,chromato_cube,mod_time,time_rn,rt1_delta,rt2_delta):
    intensity_values_list = []
    for i, coordinate in enumerate(coordinates):
        int_values = mass_spec.read_spectrum_from_chromato_cube(coordinate, chromato_cube=chromato_cube)
        rt_values= projection.matrix_to_chromato(coordinates[[i]], time_rn, mod_time, chromato_cube[0,:,:].shape)
        intensity_values_list.append(np.concatenate(([rt_values[0,0]*60,rt_values[0,1]], int_values)))

    intensity_values_list = np.array(intensity_values_list)
    rt_vals = intensity_values_list[:, :2]  # shape (n, 2)
    spectra = intensity_values_list[:, 2:]  # shape (n, m)
    
    # Compute cosine distances in vectorized form
    spec_dists = cosine_distances(spectra)
    rt_penalty_value= rt_penalty(rt_vals,rt1_delta,rt2_delta)
    distance_matrix = spec_dists + 0.01 *rt_penalty_value
    return distance_matrix


def cluster_peak(distance_matrix,chromato,coordinates, thr_debscan,min_sample_db_scan):
    clustering = DBSCAN(eps=thr_debscan, min_samples=min_sample_db_scan, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Ignore noise
    ncluster=len(unique_labels)
    clusters = []
    for i in range(ncluster):
        clusters.append([])
    for i, (t1, t2) in enumerate(coordinates):
        clusters[clustering.labels_[i]].append([t1, t2])
    coordinates = []
    for cluster in clusters:
        if (len(cluster) > 1):
            coord = cluster[np.argmax(np.array([chromato[coord[0], coord[1]] for coord in cluster]))]
        else:
            coord = cluster[0]
        coordinates.append(coord)
    coordinates = np.array(coordinates)
    return coordinates, clusters

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
        blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)
        blobs_dog = blobs_dog.astype(int)
        blobs_dog = np.array(blobs_dog)
        coordinates, radius = np.delete(blobs_dog, 2, -1), blobs_dog[:, 2]
        return coordinates


def detect_peak_dog_mp(chromato_cube,
                       abs_threshold=500,
                       rel_threshold=0.00001,
                       noise_factor=3,
                       min_sigma=1, max_sigma=20,
                       sigma_ratio=2,
                       overlap=0.5,parallel=True):

    inputs = range(chromato_cube.shape[0])
    results=[]
    if(parallel):
        num_workers = min(multiprocessing.cpu_count(),64)
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


def cluster_per_mass(coordinate,chromato_cube,time_rn):
    coordinate_cluster=[]
    for mass, coord in enumerate(coordinate) :
        tmp=chromato_cube[mass,:,:]
        npeak=len(coord)
        if(npeak!=0):  
                distance_matrix=compute_distance_metric(coord,chromato_cube,1.7,time_rn,rt1_delta=5, rt2_delta=0.1)
                coordinates, clusters= cluster_peak(distance_matrix,tmp,coord,thr_debscan=0.05,min_sample_db_scan=1)
                npeak= len(coordinates)
                if npeak>600:
                    intensities = np.array([tmp[coord[0], coord[1]] for coord in coordinates])
                    top_indices = np.argsort(intensities)[-600:][::-1]  # descending order
                    coordinates = coordinates[top_indices]
                res=[]
                for x,y in coordinates:
                    res.append([mass,x,y])
                coordinate_cluster.append(res)
    return coordinate_cluster



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