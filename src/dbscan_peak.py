from peak_detection import peak_detection
import projection
import numpy as np
import mass_spec
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from skimage.restoration import estimate_sigma
from sklearn.metrics.pairwise import cosine_distances
import multiprocessing
from multiprocessing import Pool
import time
import skimage
from peak_detection import intensity_threshold_decision_rule

#clusturing
#penlaty rt1 et 2 pour l'laignement intra mass
#seuil db scan intra mass
#penlaty rt1 et 2 pour l'laignement inter mass
#seuil db scan inter mass

#detetcion de pic
#method de detection : savitzky golay filter (peak_local_max); persistent homology param
#seuil relatif par mass 
#abs seuil (dependant du chorma?)
#nb max pic par mass
#releatif seuil 


# Simple function to test parallelism
def square(x,y):
    print(f"Processing {x} in PID {multiprocessing.current_process().pid}")
    time.sleep(1)  # Simulate work
    return x * y

def multi_process_test(x,y):
    inputs = range(x)  # List of inputs
    num_workers = 4          # Number of parallel processes
    start = time.time()
    res=[]
    with multiprocessing.Pool(processes = num_workers) as pool:
        for i, result in enumerate(pool.starmap(square, [(i,y) for i in inputs])):
            res.append(result)
    print("Results:", res)
    print("Total time:", round(time.time() - start, 2), "seconds")
    return res

def detection_mass_par_mass(chromato_cube,chromato, time_rn, mass_range):
    inputs = range(chromato_cube.shape[0])
    num_workers = 4
    results=[]
    #if(parralel):
    #    with multiprocessing.Pool(processes = num_workers) as pool:
    #        for i, result in enumerate(pool.starmap(process_mass, [(mass,chromato_cube,time_rn, mass_range) for mass in inputs])):
    #            results.append(result)
    #without parralele 
    print("start peak detection")
    for mass in inputs:
        results.append(process_mass(mass,chromato_cube,time_rn, mass_range ))
    
    print("compute distance metric")
    results = [res for res in results if res is not None]
    coordinates_all_mass=[]
    for elt in results:
        for x,y,z in elt:
            coordinates_all_mass.append([x,y,z])
    
    coordinates_all_mass = np.delete(coordinates_all_mass, 0, -1)
    
    intensity_values_list = []
    for i, coordinate in enumerate(coordinates_all_mass):
        int_values = mass_spec.read_spectrum_from_chromato_cube(coordinate, chromato_cube=chromato_cube)
        rt_values= projection.matrix_to_chromato(coordinates_all_mass[[i]], time_rn, 1.7, chromato.shape)
        intensity_values_list.append(np.concatenate(([rt_values[0,0]*60,rt_values[0,1]], int_values)))

    intensity_values_list = np.array(intensity_values_list)
    rt_vals = intensity_values_list[:, :2]  # shape (n, 2)
    spectra = intensity_values_list[:, 2:]  # shape (n, m)
    
    # Compute cosine distances in vectorized form
    spec_dists = cosine_distances(spectra)
    rt_penalty_value= rt_penalty(rt_vals,rt1_delta=2, rt2_delta=0.02 )
    distance_matrix = spec_dists + 0.01 *rt_penalty_value
    
    print("start clusturing")
    coordinates, clusters= cluster_peak(distance_matrix,chromato,coordinates_all_mass,thr_debscan=0.02,min_sample_db_scan=2)
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
    return coordinates, cluster

def process_mass(mass,chromato_cube,time_rn, mass_range):
    tmp= chromato_cube[mass,:,:]
    tmp= savgol_filter(tmp,  100, 2, mode='nearest')
    tmp[tmp<0]=0
    sigma = estimate_sigma(tmp, channel_axis=None)

    MIN_SEUIL = min(5 * sigma * 100 / np.max(tmp),0.01)
    # detect peaks
    intensity_threshold = intensity_threshold_decision_rule(
            abs_threshold=1500, rel_threshold=0.0000001, noise_factor=3, sigma=sigma, chromatogram=tmp)
    coordinates = skimage.feature.peak_local_max(
            tmp,
            min_distance=1,
            threshold_abs=intensity_threshold)
    npeak= len(coordinates)

    if npeak>600:
        intensity_threshold = intensity_threshold_decision_rule(
            abs_threshold=1500, rel_threshold=0.01, noise_factor=2, sigma=sigma, chromatogram=tmp)
        MIN_SEUIL= intensity_threshold / np.max(tmp)
        coordinates = skimage.feature.peak_local_max(
            tmp,
            min_distance=1,
            threshold_abs=intensity_threshold)
        npeak= len(coordinates)

    if(npeak!=0):  
        coordinates_in_chromato=projection.matrix_to_chromato(coordinates, time_rn, 1.7, tmp.shape)
        intensity_values_list = []
        for i, coordinate in enumerate(coordinates):
            int_values = mass_spec.read_spectrum_from_chromato_cube(coordinate, chromato_cube=chromato_cube)
            rt_values= projection.matrix_to_chromato(coordinates[[i]], time_rn, 1.7, tmp.shape)
            intensity_values_list.append(np.concatenate(([rt_values[0,0]*60,rt_values[0,1]], int_values)))

        intensity_values_list = np.array(intensity_values_list)

        # Separate RT and spectrum
        rt_vals = intensity_values_list[:, :2]  # shape (n, 2)
        spectra = intensity_values_list[:, 2:]  # shape (n, m)

        # Compute cosine distances in vectorized form
        spec_dists = cosine_distances(spectra)

        # Compute RT penalty (broadcasted)
        rt_penalty_value= rt_penalty(rt_vals,rt1_delta=5, rt2_delta=0.1 )
        distance_matrix = spec_dists + 0.01 *rt_penalty_value

        # Apply DBSCAN with precomputed distance matrix
        coordinates, clusters= cluster_peak(distance_matrix,tmp,coordinates,thr_debscan=0.05,min_sample_db_scan=1)
        
        #ncluster=len(clusters)
        #coordinates_in_chromato_new=projection.matrix_to_chromato(coordinates, time_rn, 1.7, chromato.shape)
        #fig1=visualizer2(((tmp), time_rn), title="mass " + str(mass+mass_range[0]), log_chromato=True,  mod_time=1.7,points=coordinates_in_chromato)
        #fig2=visualizer2(((tmp), time_rn), title= str(npeak-ncluster) + " clustered peaks "+ str(round(MIN_SEUIL,3)), log_chromato=True,  mod_time=1.7,points=coordinates_in_chromato_new)
        
        res=[]
        for x,y in coordinates:
            res.append([mass,x,y])

        return res
    
    #fig1=visualizer2(((tmp), time_rn), title="mass " + str(mass+mass_range[0]), log_chromato=True,  mod_time=1.7)
    return None

