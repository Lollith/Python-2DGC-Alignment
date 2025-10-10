# import profile
import read_chroma
import baseline_correction
import peak_detection
import matching

import csv
import numpy as np
import os
import traceback
import time
import projection
import plot

import dbscan_peak
import h5py
import imagepers
from projection import chromato_to_matrix, matrix_to_chromato
import uuid
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d


def write_line(compound_name, rt1, rt2, area, formatted_spectrum):
    return (compound_name + "\t" + "\"" + str(rt1 * 60) + " , " + str(rt2)
            + "\"" + "\t" + str(area) + "\t" + "T" + "\t" + formatted_spectrum
            + "\n")


def mass_spectra_format(mass_range, int_values):
    """
    Formats the mass spectrum data into a string representation, sorting the
    mass-to-intensity pairs by intensity in descending order. Each pair is
    represented as "mass:intensity" and separated by spaces.

    Parameters:
    -----------
    mass_range : tuple of int
        A tuple representing the range of mass values, in the form (min_mass,
        max_mass),
        where the mass values should be within this range.

    int_values : ndarray
        A numpy array containing the intensity values corresponding to the
        mass values in `mass_range`.

    Returns:
    --------
    str
        A formatted string where each mass-to-intensity pair is represented as
        "mass:intensity", with pairs sorted in descending order of intensity.

    Example:
    --------
    >>> mass_range = (100, 110)
    >>> int_values = np.array([10, 20, 15, 25, 30, 5, 40, 10, 12, 8])
    >>> result = mass_spectra_format(mass_range, int_values)
    >>> print(result)
    "110:40 109:30 108:25 107:20 106:15 105:12 104:10 103:10 102:8 101:5"
    """

    range_min, range_max = mass_range
    mass_values = np.linspace(range_min, range_max, len(int_values)).astype(int)
    spectrum = np.column_stack((mass_values, int_values))
    sorted_by_int_spectrum = spectrum[(-spectrum[:, 1]).argsort()]
    formatted_spectrum = ""
    for i, mz_int in enumerate(sorted_by_int_spectrum):
        if (i != 0):
            formatted_spectrum = formatted_spectrum + " "
        formatted_spectrum = (formatted_spectrum + str(mz_int[0]) + ":" +
                              str(mz_int[1]))
    return formatted_spectrum


# def find_peak_bounds_valley_detection(profile, peak_idx):
#     """
#     Valley detection avec validation du ratio baseline/peak
#     """
#     peak_intensity = profile[peak_idx]
#     baseline = np.percentile(profile, 10)
#     peak_height = peak_intensity - baseline
    
#     # Valley detection classique
#     min_prominence = peak_height * 0.25
#     smooth_profile = gaussian_filter1d(profile, sigma=1)
#     inverted = -smooth_profile
#     valleys, _ = find_peaks(inverted, prominence=min_prominence)
    
#     # Trouver vallées proches
#     left_valleys = valleys[valleys < peak_idx]
#     right_valleys = valleys[valleys > peak_idx]
    
#     # VALIDATION : Vérifier que les zones entre vallées ne contiennent pas d'autres gros pics
#     left_bound = left_valleys[-1] if len(left_valleys) > 0 else max(0, peak_idx - 20)
#     right_bound = right_valleys[0] if len(right_valleys) > 0 else min(len(profile) - 1, peak_idx + 20)
    
#     # Extraire la région entre les bornes
#     region = profile[left_bound:right_bound+1]
#     region_max = np.max(region)
#     region_median = np.median(region)
    
#     print(f"🔍 Valley Validation:")
#     print(f"   Region max: {region_max:.0f}")
#     print(f"   Region median: {region_median:.0f}")
#     print(f"   Peak: {peak_intensity:.0f}")
    
#     # Si la région contient des intensités très élevées en dehors du pic principal, c'est suspect
#     region_without_peak = np.concatenate([
#         region[:peak_idx-left_bound], 
#         region[peak_idx-left_bound+1:]
#     ])
    
#     if len(region_without_peak) > 0:
#         second_highest = np.max(region_without_peak)
#         ratio = second_highest / peak_intensity
        
#         print(f"   Second highest in region: {second_highest:.0f} (ratio: {ratio:.2f})")
        
#         # Si il y a un autre pic > 60% du pic principal, rejeter
#         if ratio > 0.6:
#             print(f"   ❌ Multi-peak detected - ratio {ratio:.2f} > 0.6")
#             raise ValueError("Multiple peaks in region")
    
#     return left_bound, right_bound

# def find_peak_bounds_valley_detection(profile, peak_idx, min_prominence=0.3):
#     """
#     Trouve les bornes en dÃ©tectant les vallÃ©es autour du pic
    
#     Parameters:
#     -----------
#     profile : array
#         Profil d'intensitÃ© 1D
#     peak_idx : int
#         Index du maximum du pic
#     min_prominence : float
#         Prominence minimale pour dÃ©tecter une vallÃ©e
        
#     Returns:
#     --------
#     left_bound, right_bound : int, int
#     """

#     peak_intensity = profile[peak_idx]
#     baseline = np.percentile(profile, 10)
#     peak_height = peak_intensity - baseline
    
#     # PROMINENCE ADAPTÉE au pic (pas fixe !)
#     min_prominence = peak_height * min_prominence  # 30% de la hauteur du pic
    
#     print(f"🔍 Valley Detection Fixed:")
#     print(f"   Peak height: {peak_height:.0f}")
#     print(f"   Min prominence: {min_prominence:.0f} ({min_prominence*100}%)")
    
#     # Lisser lÃ©gÃ¨rement pour Ã©viter les micro-oscillations
#     smooth_profile = gaussian_filter1d(profile, sigma=1)
    
#     # Inverser le signal pour dÃ©tecter les minima comme des maxima
#     inverted = -smooth_profile
    
#     # DÃ©tecter tous les minima (vallÃ©es)
#     valleys, _ = find_peaks(inverted, prominence=min_prominence)
#     if len(valleys) == 0:
#         raise ValueError("No valleys found")
    
#     # Trouver la vallÃ©e la plus proche Ã  gauche
#     left_valleys = valleys[valleys < peak_idx]
#     left_bound = left_valleys[-1] if len(left_valleys) > 0 else 0
    
#     # Trouver la vallÃ©e la plus proche Ã  droite  
#     right_valleys = valleys[valleys > peak_idx]
#     right_bound = right_valleys[0] if len(right_valleys) > 0 else len(profile) - 1
    
#     return left_bound, right_bound

# def find_peak_bounds_robust(profile, peak_idx, 
#                            fwhm_factor=0.6, min_width=5):
#     """
#     MÃ©thode hybride combinant plusieurs approches
    
#     1. Valley detection (prioritaire)
#     2. Baseline threshold (fallback)

#     """
#     peak_intensity = profile[peak_idx]
    
#     # === MÃ‰THODE 1: Valley Detection ===
#     try:
#         left_valley, right_valley = find_peak_bounds_valley_detection(profile, peak_idx)
        
#         # VÃ©rifier que les bornes sont raisonnables
#         width = right_valley - left_valley
#         if width >= min_width and width <= len(profile) // 2:
#             return left_valley, right_valley
#     except:
#         pass
    
#     # === MÃ‰THODE 2: Baseline Threshold (Fallback) ===
#     print("   ⚠️ Valley detection failed or unreasonable, using baseline threshold...")
#     baseline = np.percentile(profile, 10)  # 10% percentile comme baseline
#     half_height = baseline + (peak_intensity - baseline) / 2
#     threshold = half_height * (1 - fwhm_factor)
    
#     # Borne gauche
#     left_bound = peak_idx
#     for i in range(peak_idx - 1, -1, -1):
#         if profile[i] <= threshold:
#             left_bound = i + 1
#             break
#         left_bound = i
    
#     # Borne droite
#     right_bound = peak_idx  
#     for i in range(peak_idx + 1, len(profile)):
#         if profile[i] <= threshold:
#             right_bound = i - 1
#             break
#         right_bound = i
    
#     width = right_bound - left_bound + 1
#     if width < min_width:
#         expand = (min_width - width) // 2
#         left_bound = max(0, left_bound - expand)
#         right_bound = min(len(profile) - 1, right_bound + expand)
    
#     return left_bound, right_bound


# def has_clear_valleys(roi):
#     """
#     Vérifie s'il y a des vallées significatives dans la région
#     """
    
#     # Lisser légèrement
#     smooth_roi = gaussian_filter1d(roi, sigma=1)
    
#     # Chercher des vallées (minima)
#     inverted = -smooth_roi
#     valleys, properties = find_peaks(inverted, prominence=np.std(roi) * 2)
    
#     # Il y a des vallées claires si :
#     # 1. Au moins une vallée détectée
#     # 2. Prominence significative par rapport au bruit
#     return len(valleys) > 0 and np.max(properties['prominences']) > np.std(roi) * 1.5

# def valley_detection_in_roi(roi):
#     """
#     Applique valley detection dans la ROI seulement
#     """
#     from scipy.signal import find_peaks
    
#     # Détecter toutes les vallées
#     inverted = -roi
#     valleys, _ = find_peaks(inverted, prominence=np.std(roi))
    
#     # Trouver le pic principal dans la ROI
#     peak_idx_roi = np.argmax(roi)
    
#     # Vallées à gauche et droite
#     left_valleys = valleys[valleys < peak_idx_roi]
#     right_valleys = valleys[valleys > peak_idx_roi]
    
#     # Bornes = vallées les plus proches du pic
#     left_bound = left_valleys[-1] if len(left_valleys) > 0 else 0
#     right_bound = right_valleys[0] if len(right_valleys) > 0 else len(roi) - 1
    
#     return left_bound, right_bound

# def chromatof_bounds_exact(profile, peak_idx, expected_width, snr):
#     """
#     Reproduction exacte de l'algorithme ChromaTOF LECO
#     """
#     peak_intensity = profile[peak_idx]
#     # Fenêtre de recherche baseline (large)
#     search_window = expected_width * 3
#     start = max(0, peak_idx - search_window)
#     end = min(len(profile), peak_idx + search_window)
    
#     # Baseline = minimum absolu dans la fenêtre
#     baseline = np.min(profile[start:end])
    
#     # Hauteur et seuil
#     peak_height = profile[peak_idx] - baseline
#     threshold = baseline + (peak_height / snr)
    
#     # Limites de recherche des bornes
#     max_left = max(0, peak_idx - expected_width)
#     max_right = min(len(profile) - 1, peak_idx + expected_width)

#     print(f"🔍 DEBUG ChromaTOF:")
#     print(f"   Peak: {peak_intensity:.0f}, Baseline: {baseline:.0f}")
#     print(f"   Threshold: {threshold:.0f} (SNR={snr})")
#     print(f"   Search zone: [{max_left}:{max_right}] (±{expected_width})")
#     print(f"   Profile in zone: {profile[max_left:max_right+1]}")
    
#     # Recherche borne gauche
#     left_bound = peak_idx
#     for i in range(peak_idx, max_left - 1, -1):
#         if profile[i] <= threshold:
#             left_bound = i + 1
#             break
#         left_bound = i
    
#     # Recherche borne droite
#     right_bound = peak_idx  
#     for i in range(peak_idx, max_right + 1):
#         if profile[i] <= threshold:
#             right_bound = i - 1
#             break
#         right_bound = i
    
#     return left_bound, right_bound

def method_chromatof(profile, peak_idx, snr=2):
    """
    ChromaTOF SANS contrainte d'expected_width
    """
    peak_intensity = profile[peak_idx]
    
    # Baseline sur une grande fenêtre
    search_window = 50
    start = max(0, peak_idx - search_window)
    end = min(len(profile), peak_idx + search_window)
    baseline = np.percentile(profile[start:end], 5)
    
    peak_height = peak_intensity - baseline
    threshold = baseline + (peak_height / snr)
    
    print(f"🔧 ChromaTOF Unconstrained:")
    print(f"   Peak: {peak_intensity:.0f}, Baseline: {baseline:.0f}")
    print(f"   SNR: {snr} → Threshold: {threshold:.0f}")
    print(f"   Threshold is {threshold/peak_intensity*100:.1f}% of peak")
    
    left_bound = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if profile[i] <= threshold:
            left_bound = i + 1
            break
        left_bound = max(0, i)
    
    right_bound = peak_idx
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] <= threshold:
            right_bound = i - 1
            break
        right_bound = min(len(profile) - 1, i)
    
    width = right_bound - left_bound + 1
    print(f"   Result: [{left_bound}:{right_bound}] = {width} points")
    
    return left_bound, right_bound

def find_peak_bounds_intelligent(profile, peak_idx):
    """
    Méthode intelligente qui s'adapte au type de pic
    """
    peak_intensity = profile[peak_idx]
    global_baseline = np.percentile(profile, 5)
    peak_height = peak_intensity - global_baseline
    
    print(f"🧠 Intelligent Peak Analysis:")
    print(f"   Peak: {peak_intensity:.0f}, Global baseline: {global_baseline:.0f}")
    print(f"   Peak height: {peak_height:.0f}")
    
    # === DIAGNOSTIC DU TYPE DE PIC ===
    
    # 1. Tester d'abord avec percentage standard (5%)
    threshold_low = global_baseline + peak_height * 0.05

    left_test = peak_idx
    for i in range(peak_idx - 1, max(0, peak_idx - 100), -1):
        if profile[i] <= threshold_low:
            left_test = i + 1
            break
        left_test = i
    
    right_test = peak_idx
    for i in range(peak_idx + 1, min(len(profile), peak_idx + 100)):
        if profile[i] <= threshold_low:
            right_test = i - 1
            break
        right_test = i
    
    width_low = right_test - left_test + 1
    
    print(f"   Test 5%: threshold={threshold_low:.0f} → width={width_low}")
    
    # 2. Détecter le type de pic
    if width_low > 100:  # Pic problématique
        print(f"   → PROBLEMATIC PEAK detected (width={width_low} > 100)")
        
        # Analyser la zone autour du pic pour baseline locale
        window = 30
        left_zone = max(0, peak_idx - window)
        right_zone = min(len(profile), peak_idx + window)
        
        # Prendre les minima aux extrémités
        left_min = np.min(profile[left_zone:peak_idx-10]) if peak_idx > 10 else global_baseline
        right_min = np.min(profile[peak_idx+10:right_zone]) if peak_idx < len(profile)-10 else global_baseline
        local_baseline = max(left_min, right_min, global_baseline)  # Le plus élevé des trois
        
        local_height = peak_intensity - local_baseline
        
        # Utiliser percentage élevé avec baseline locale
        # for pct in [0.30, 0.40, 0.50, 0.60]:
        #     threshold = local_baseline + local_height * pct
            
        #     left_bound = peak_idx
            # for i in range(peak_idx - 1, -1, -1):
            #     if profile[i] <= threshold:
            #         left_bound = i + 1
            #         break
            #     left_bound = max(0, i)
            
            # right_bound = peak_idx
            # for i in range(peak_idx + 1, len(profile)):
            #     if profile[i] <= threshold:
            #         right_bound = i - 1
            #         break
            #     right_bound = min(len(profile) - 1, i)
            
            # width = right_bound - left_bound + 1

            # aire = np.trapz(profile[left_bound:right_bound+1])
            # aire_ref = np.trapz(profile[max(0,peak_idx-40):min(len(profile), peak_idx+40)])
            
            # if 15 <= width <= 60:  # Largeur acceptable
            #     print(f"   → Using {pct*100}% with local baseline: [{left_bound}:{right_bound}] = {width}")
            #     return left_bound, right_bound
        best = None
        for pct in [0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0]:
            threshold = local_baseline + local_height * pct
            left_bound = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if profile[i] <= threshold:
                    left_bound = i + 1
                    break
                left_bound = max(0, i)

            right_bound = peak_idx
            for i in range(peak_idx + 1, len(profile)):
                if profile[i] <= threshold:
                    right_bound = i - 1
                    break
                right_bound = min(len(profile) - 1, i)

            # calcul des bornes comme déjà
            width = right_bound - left_bound + 1
            if best is not None and best[2] < 0.75*width:
                print(f"→ Fallback: étend forçage de la fenêtre [-80, +80] ind autour du pic")
                left_bound = max(0, peak_idx-80)
                right_bound = min(len(profile)-1, peak_idx+80)
                print(f"→ Fallback: utilise la largeur visuelle à 5% → [{left_test}:{right_test}]")
                return left_bound, right_bound
            if best is not None:
                print(f"→ Using {best[4]*100:.3f}% best matching: [{best[0]}:{best[1]}] = {best[2]}, aire={best[3]:.1f}")
                return best[0], best[1]
        
        # Fallback si rien ne marche
        n = int(width_low / 2)
        left_bound = max(0, peak_idx - n)
        right_bound = min(len(profile) - 1, peak_idx + n)
        print(f"   → Forcing fixed width: [{left_bound}:{right_bound}]")
        print(f"Test pct={pct}: width={width}")
        print(f"Largeur base visuelle (5%): {width_low}")
        return left_bound, right_bound
        
    else:  # Pic normal
        print(f"   → NORMAL PEAK detected (width={width_low} ≤ 100)")
        
        # Utiliser percentage standard optimisé
        if 10 <= width_low <= 60:
            print(f"   → Using 5%: [{left_test}:{right_test}] = {width_low}")
            return left_test, right_test
        else:
            # Ajuster légèrement le percentage
            for pct in [0.08, 0.10, 0.12, 0.15]:
                threshold = global_baseline + peak_height * pct
                
                left_bound = peak_idx
                for i in range(peak_idx - 1, -1, -1):
                    if profile[i] <= threshold:
                        left_bound = i + 1
                        break
                    left_bound = max(0, i)
                
                right_bound = peak_idx
                for i in range(peak_idx + 1, len(profile)):
                    if profile[i] <= threshold:
                        right_bound = i - 1
                        break
                    right_bound = min(len(profile) - 1, i)
                
                width = right_bound - left_bound + 1
                
                if 10 <= width <= 50:
                    print(f"   → Using {pct*100}%: [{left_bound}:{right_bound}] = {width}")
                    return left_bound, right_bound
            
            # Fallback
            print(f"   → Using original 5% result: [{left_test}:{right_test}]")
            return left_test, right_test



def compute_matches_identification(matches, sepc_list, area,chromato, chromato_cube,time_rn,mod_time,
                                   mass_range, sample_name, formated_spectra,
                                   quant="mass",extract_patch=False,output_hdf5_file=None,
                                   integration_mod_max1=True, integration_mod_max2=False,
                                   similarity_threshold=0.001
                                   ):
    """
    Computes the identification data for each match in a list of matches,
    including the integration of peak areas and heights, along with additional
    compound information. Optionally formats the spectra associated with each
    match.

    Parameters:
    -----------
    matches : list of tuples
        A list of matches, where each match is a tuple containing:
        - (RT1, RT2): The retention times of the match.
        - A dictionary containing compound data such as 'casno',
        'compound_name', 'compound_formula',
          'hit_prob', 'match_factor', 'reverse_match_factor', and 'spectra'.
        - A coordinate tuple (coord) representing the position of the match.

    chromato : ndarray
        A 2D array representing the chromatogram data with intensity values
        for each point.

    chromato_cube : ndarray
        A 3D array representing the chromatogram cube, containing spectral
        data for each chromatogram point.

    mass_range : tuple of int
        A tuple representing the range of mass values (min_mass, max_mass) for
        the spectrum formatting.

    similarity_threshold : float, optional, default=0.001
        The threshold for similarity when checking peak pool similarity. A
        lower value will result in stricter matching criteria.

    formated_spectra : bool, optional, default=False
        If True, formats and includes the mass spectra in the identification
        data for each match.

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing the identification data for a
        match, including:
        - 'casno', 'compound_name', 'compound_formula', 'hit_prob',
        'match_factor', 'reverse_match_factor', 'rt1', 'rt2', 'area', 'height',
        and optionally 'spectra' (if `formated_spectra` is True).

    Example:
    --------
    >>> matches = [( (5.2, 5.3), {'casno': '123-45-6', 'compound_name':
        'Compound A', 'compound_formula': 'C6H12O6',
        'hit_prob': 0.95, 'match_factor': 0.98, 'reverse_match_factor': 0.97,
        'spectra': [100, 200, 150]}, (3, 4))]
    >>> chromato = np.array([[0, 1], [2, 3]])
    >>> chromato_cube = np.random.rand(10, 2, 2)  # Example 3D data
    >>> mass_range = (50, 150)
    >>> result = compute_matches_identification(matches, chromato,
        chromato_cube, mass_range)
    >>> print(result)
    """
    # print("compute matches , chromatocube shape:", chromato_cube.shape)
    matches_identification = []
    sample_metadata_list =[]
    max_len = max(len(match) for match in matches)

    # ComplÃ©ter les lignes plus courtes ou tronquer les lignes trop longues
    matches = [match + [None] * (max_len - len(match)) if len(match) < max_len else match[:max_len] for match in matches]
    matches = np.array(matches, dtype=object)
    min_mz, max_mz = int(mass_range[0]), int(mass_range[1])
    canonical_length = max_mz - min_mz + 1
    basename = os.path.splitext(sample_name)[0]
    sample_name_group = basename
    if extract_patch: 
        with h5py.File(output_hdf5_file, "a") as h5_file:
            if sample_name_group not in h5_file:
                sample_group_h5 = h5_file.create_group(sample_name_group)

    for j, match in enumerate(matches):
        
        match_data_list = match[1] \
            if isinstance(match[1], list) else [match[1]]
        
        coord = match[2]
        spectrum_data = match[1][0]['spectra']
        spec_deconvo = sepc_list[j]
        majority_mass = np.argmax(spec_deconvo)
        
        if(quant == "mass"):
            chromato_m = chromato_cube[majority_mass,: ,: ] ## pas sur le chromato mais sur la masse majoritaire 
        else: 
            chromato_m=chromato
        
        # TODO tester
        # FWHM = 0.04
        # FWHM_FACTOR = 0.85
        SNR = 2
        if(integration_mod_max1):
            # print("methode nicolas")
            # blob = integration.peak_pool_similarity_check(
            #         chromato_m, np.stack(matches[:, 2]), coord, chromato_cube,
            #         threshold=0.01, plot_labels=True,
            #         similarity_threshold=similarity_threshold)                
            # area_total = integration.compute_area(chromato_m, blob) 

            # mask = np.zeros_like(blob)
            # mask[coord[0],: ] = blob[coord[0],:]
            # area_mod_max = integration.compute_area(chromato_m, mask)
            # print("methode chromatof_valley")
            # rt2_hz = chromato.shape[1] / mod_time                    # Fréquence RT2
            # baseline_width_sec = FWHM * 2.35                  # Largeur de base en secondes
            # expected_width = int(baseline_width_sec * rt2_hz)        # Points attendus
    
            # # print(f"🎯 RT2: {rt2_hz:.0f} Hz → Expected width: {expected_width} points")
            # #snr =3 = standard chromatof
            # left, right = method_chromatof(chromato_m[coord[0], :], coord[1], expected_width, snr=SNR)
            # area_mod_max = np.sum(chromato_m[coord[0], left:right+1])
            # VISUALISATION (pour les premiers pics seulement)
            # if j < 20:  # Afficher seulement les 10 premiers pics
            #     results = visualize_integration_methods(
            #         chromato_m, coord, mod_time, FWHM, peak_name=f"Peak {j+1}", FWHM_FACTOR=FWHM_FACTOR, SNR=SNR
            #     )
            print(f"Peak at {coord} → Bounds: ({left}, {right}), Area: {area_mod_max}")

        #TODO    
        elif integration_mod_max2:
            print("methode valley")
            left_bound, right_bound = find_peak_bounds_simple_percentage(chromato_m[coord[0], :], coord[1])
            # left_bound, right_bound = find_peak_bounds_robust(chromato_m[coord[0], :], coord[1], fwhm_factor=FWHM_FACTOR)
            area_mod_max = np.sum(chromato_m[coord[0], left_bound:right_bound+1])
            # VISUALISATION
            # if j < 5:  # Afficher seulement les 5 premiers pics
            #     results = visualize_integration_methods(
            #         chromato_m, coord, mod_time, FWHM=0.04, peak_name=f"Peak {j+1}"
            #     )
            print(f"Peak at {coord} → Bounds: ({left_bound}, {right_bound}), Area: {area_mod_max}")

        else:
            area_mod_max = 0

        height = chromato_m[coord[0], coord[1]]
        area_j = area[j]

        def join_field(field):
            return '/'.join(str(m.get(field, '')) for m in match_data_list)

        identification_data_dict = {
            'compound_name': join_field('compound_name'),
            'casno': join_field('casno'),
            'compound_formula': join_field('compound_formula'),
            'hit_prob': join_field('hit_prob'),
            'match_factor': join_field('match_factor'),
            'reverse_match_factor': join_field('reverse_match_factor'),
            'rt1': match[0][0],
            'rt2': match[0][1],
            'area': area_j,
            'area_mod_max':area_mod_max,
            'height': height,
            "quant_mass":majority_mass + mass_range[0]
        }

        if formated_spectra:
            # print("match =", match)
            # print("match[1] =", match[1], type(match[1]))
            sample_spectra = match[1][0].get('spectra', None) 
            identification_data_dict['spectra'] = mass_spectra_format(mass_range, sample_spectra)

            identification_data_dict['spectra_deconvo'] = mass_spectra_format(mass_range, spec_deconvo)

        if extract_patch:
            # Adjust spectrum length to match the canonical length for this sample
            if spectrum_data is not None:
                current_length = len(spectrum_data)
                if current_length != canonical_length:
                    if current_length < canonical_length:
                        pad_width = canonical_length - current_length
                        spectrum_data = np.pad(spectrum_data, (0, pad_width), mode='constant', constant_values=0)
                    else: # current_length > canonical_length
                        spectrum_data = spectrum_data[:canonical_length]
                      # Verify final length
                if len(spectrum_data) != canonical_length:
                           raise RuntimeError(f"Spectrum length adjustment failed! Got {len(spectrum_data)}, expected {canonical_length}")
            else: print("Warning: Spectrum extraction returned None.")
            
            context_patch={'rt1_window_minutes': 0.5,
                       'rt2_window_seconds': 0.2}
            
            chromato_shape=chromato.shape
            context_patch_data = np.array([]) # Init empty 
            rt1_min = identification_data_dict["rt1"] - context_patch['rt1_window_minutes']; rt1_max = identification_data_dict["rt1"] + context_patch['rt1_window_minutes']
            rt2_min = identification_data_dict["rt2"] - context_patch['rt2_window_seconds']; rt2_max = identification_data_dict["rt2"] + context_patch['rt2_window_seconds']
            full_rt_bounds = ((rt1_min, rt2_min), (rt1_max, rt2_max))
            position = np.array([[rt1_min, rt2_min], [rt1_max, rt2_max]])
            try: # Wrap conversion in try-except
                window_idx = chromato_to_matrix(position, time_rn, mod_time, chromato_shape)
                if window_idx is None or window_idx.shape != (2, 2): raise ValueError("Invalid window_idx shape")
            except Exception as e: raise ValueError(f"chromato_to_matrix error: {e}") from e

            x_min = int(max(np.floor(window_idx[0][0]), 0)); y_min = int(max(np.floor(window_idx[0][1]), 0))
            x_max = int(min(np.ceil(window_idx[1][0]), chromato_shape[0])); y_max = int(min(np.ceil(window_idx[1][1]), chromato_shape[1]))

            # --- Calculate mass_index using range_min ---
            mass=majority_mass+ mass_range[0]
            mass_index = majority_mass # Index relative to the start of cube's axis
 
            if mass_index < 0 or mass_index >= chromato_cube.shape[0]:
                raise ValueError(f"Mass index {mass_index} (mass {mass}, min_mz {mass_range[0]}) out of cube bounds[0]={chromato_cube.shape[0]}.")
            if x_min >= x_max or y_min >= y_max:
                print(f"Warning: Invalid pixel indices [{x_min}:{x_max}, {y_min}:{y_max}]. Returning empty patch.")
                return np.array([[]]), full_rt_bounds, mass_index

            full_patch = chromato_cube[mass_index, x_min:x_max, y_min:y_max]
            if full_patch.size == 0: print(f"Warning: Extracted full_patch empty [{x_min}:{x_max}, {y_min}:{y_max}].")
            context_patch_data=full_patch
            identification_data_dict['context_patch'] = full_patch
            identification_data_dict['full_rt_bounds'] = full_rt_bounds
            center_rt_corr =  match[0] 
            clipped_patch_data = clip_patch_by_rt(
                      context_patch_data, full_rt_bounds, center_rt_corr,
                      0.1, 0.1175 )

            unique_id_data = str(uuid.uuid4())
            clip_patch_id = f"clip_patch_{unique_id_data}" if clipped_patch_data.size > 0 else None
            context_patch_id = f"context_patch_{unique_id_data}" if context_patch_data.size > 0 else None
            spectrum_id = f"spectrum_{unique_id_data}" if spectrum_data is not None and spectrum_data.size > 0 else None
 
            with h5py.File(output_hdf5_file, "r+") as h5_file:
                sample_group_h5 = h5_file[sample_name_group]
                if clip_patch_id: sample_group_h5.create_dataset(clip_patch_id, data=clipped_patch_data, compression="gzip")
                if context_patch_id: sample_group_h5.create_dataset(context_patch_id, data=context_patch_data, compression="gzip")
                if spectrum_id: sample_group_h5.create_dataset(spectrum_id, data=spectrum_data, compression="gzip") 

                    # --- Collect Metadata ---
            metadata = {
                        "unique_id": unique_id_data,
                        "Sample": sample_name_group, "Mol": identification_data_dict['compound_name'], "mass": mass, "Area": area_j,
                        "RT1_theoretical": identification_data_dict['rt1'], "RT2_theoretical": identification_data_dict['rt2'],
                        "RT1_corrected": identification_data_dict['rt1'], "RT2_corrected": identification_data_dict['rt2'],
                        "signal_noise_ratio": None, # Store calculated SNR
                        "canonical_min_mz": min_mz, "canonical_max_mz": max_mz, "canonical_length": canonical_length,
                        "clip_patch_id": clip_patch_id, "context_patch_id": context_patch_id, "spectrum_id": spectrum_id,
                        "patch_image": None
                    }

            sample_metadata_list.append(metadata) 
                    # --- End Metadata ---

        matches_identification.append(identification_data_dict)
    return matches_identification , sample_metadata_list

def identification(filename,
                   output_path,
                   mod_time,
                   method, mode, noise_factor,
                   abs_threshold, rel_threshold, cluster, min_distance,
                   min_sigma, max_sigma, sigma_ratio,
                   num_sigma, formated_spectra, match_factor_min,
                   min_persistence, overlap, eps, min_samples,
                   nist, quant, extract_patch, output_hdf5_file,
                   method_baseline, plot_):
    r"""Takes a chromatogram as file and returns identified compounds.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    method : optional
        Method used to detect peaks.
    mode : optional
        Mode used to detect peaks. Can be either tic or mass_per_mass or 3D.
    filtering_factor :
        Used to compute theshold as seuil * estimated gaussian white noise.
    hit_prob_min : optional
        Filter compounds with hit_prob < hit_prob_min
    ABS_THRESHOLDS : optional
        If mode='mass_per_mass' or mode='3D', ABS_THRESHOLDS is the threshold
        relative to a slice of the 3D chromatogram or a slice of the 3D
        chromatogram.
    cluster : optional
        Whether to cluster coordinates when mode is mass_per_mass or 3D.
    min_distance : optional
        peak_local_max method parameter. The minimal allowed distance
        separating peaks. To find the maximum number of peaks, use
        min_distance=1.
    sigma_ratio : optional
        DoG method parameter. The ratio between the standard deviation of
        Gaussian Kernels used for computing the Difference of Gaussians.
    num_sigma : optional
        LoG/DoH method parameter. The number of intermediate values of
        standard deviations to consider between min_sigma (1) and max_sigma
        (30).
    formated_spectra : optional
        If spectra need to be formatted for peak table based alignment.
    match_factor_min : optional
        Filter compounds with match_factor < match_factor_min.
    -------
    Returns
    -------
    matches_identification:
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    --------
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]

    chromato_tic, time_rn, chromato_cube, sigma, mass_range = (
        read_chroma.read_chromato_and_chromato_cube(filename,
                                                    output_path,
                                                    mod_time,
                                                    pre_process=False, plot_=plot_
                                                    ))

    baseline_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube, method=method_baseline))

    if (mode == "mass_per_mass") & (method == "DoG"):
        print(f"Peak detection with mode {mode} and method {method}")
        max_peak_per_mass = 600
        coordinates, spec_list, area = dbscan_peak.detection_mass_par_mass_Dog(
            baseline_cube, (chromato_tic, time_rn),
            mod_time,
            abs_threshold,
            rel_threshold,
            noise_factor,
            min_sigma,
            max_sigma,
            sigma_ratio,
            overlap,
            max_peak_per_mass,
            rt1_delta=2,
            rt2_delta=0.02,
            min_size_cluster_mass=3,
            thr_debscan=0.04,
            multi_processing=True,
            cleaning_close_peak=True,)

        print("Peaks number: ", len(coordinates))
        if (plot_):
            print("Plotting detected peaks...")
            dir_save = f"{output_path}"
            coordinates_in_chromato = projection.matrix_to_chromato(
                    coordinates, time_rn, mod_time, chromato_tic.shape)
            fig_title = f"{base_name}#Dt_{mode}_{method}"
            plot.visualizer((chromato_tic, time_rn), mod_time,
                            title=fig_title, log_chromato=True,
                            points=coordinates_in_chromato,
                            save=True, dir_save=dir_save)

        matches = matching.matching_nist_lib_from_chromato_cube(
                (chromato_tic, time_rn, mass_range), baseline_cube,
                coordinates, mod_time, match_factor_min, nist)

        print("Matches found: ", len(matches))

        matches_identification, sample_metadata_list = compute_matches_identification(
                matches, spec_list, area, chromato_tic, baseline_cube,time_rn,mod_time, mass_range,base_name,
                formated_spectra, quant, extract_patch, output_hdf5_file)

        return matches_identification, sample_metadata_list

    elif (mode == "tic") & (method == "persistent_homology"):
        intensity_threshold = peak_detection.intensity_threshold_decision_rule(
            abs_threshold, rel_threshold, noise_factor, sigma, chromato_tic)
        g0 = imagepers.persistence(chromato_tic)
        pts = []
        for i, homclass in enumerate(g0):
            p_birth, birth_val, pers_val, p_death = homclass
            x, y = p_birth
            # Apply thresholds
            max_peak_val = np.max(chromato_tic)
            if chromato_tic[x, y] < intensity_threshold:
                continue
            if pers_val < min_persistence * max_peak_val:
                continue
            pts.append((x, y))
        
        coordinates = np.array(pts)
        print("Peaks number: ", len(coordinates))
        
        if plot_:
            coordinates_in_chromato = projection.matrix_to_chromato(
                    coordinates, time_rn, mod_time, chromato_tic.shape)
            plot.visualizer((chromato_tic, time_rn), mod_time,
                                title = f"peak detection with mode {mode} and method {method}",
                                log_chromato = True, points=coordinates_in_chromato)
    

        # return np.array(pts), []
        return coordinates, "persistent_homology_mode"
    else:
        print("Autres fonctions de detection ne fonctionnent pas pour le moment")
        return [], []
        # coordinates = peak_detection.peak_detection()

    


def cohort_identification_to_csv(filename, matches_identification, PATH):
    r"""Generate csv (readable) peak table.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    matches_identification :
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    PATH : optional
        Path to the resulting formatted peak table.

    Returns
    -------
    None
        The function writes a CSV file, containing one line per identified 
        compound:
        - Name : Chemical compound name (e.g., Toluene)
        - Casno : CAS number (unique compound identifier)
        - Formula : Molecular formula (e.g., C7H8)
        - hit_prob : Hit probability (%), confidence in the identification
        - match_factor : Match factor between observed and library spectra
        - reverse_match_factor : Reverse match factor ignoring unmatched peaks
        in the sample
        - rt1 : Retention time in the 1st dimension
        - rt2 : Retention time in the 2nd dimension
        - Area : Peak area (proportional to the compound abundance)
        - Height : Peak height (related to concentration)
    """

    with open(PATH + filename + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        # header
        writer.writerow(['Name', 'Casno', 'Formula', 'hit_prob',
                         'match_factor', 'reverse_match_factor', 'rt1', 'rt2',
                         'Area', 'Height'])

        for identification_data_dict in matches_identification:
            casno = identification_data_dict['casno']
            compound_name = identification_data_dict['compound_name']
            compound_formula = identification_data_dict['compound_formula']
            hit_prob = identification_data_dict['hit_prob']
            match_factor = identification_data_dict['match_factor']
            reverse_match_factor = \
                (identification_data_dict['reverse_match_factor'])
            rt1 = identification_data_dict['rt1']
            rt2 = identification_data_dict['rt2']
            area = identification_data_dict['area']
            height = identification_data_dict['height']
            row = [compound_name, casno, compound_formula, hit_prob,
                   match_factor, reverse_match_factor, rt1, rt2, area, height]
            writer.writerow(row)


# def cohort_identification_alignment_input_format_txt(
#         filename, matches_identification, PATH):
#     r"""Generate formatted peak table for alignment.

#     Parameters
#     ----------
#     filename :
#         Chromatogram full filename.
#     matches_identification :
#         Array of match dictionary containing casno, name, formula, spectra,
#         coordinates...
#     PATH : optional
#         Path to the resulting formatted peak table.
#     """
#     with open(PATH + filename + '.txt', 'w', encoding='UTF8') as f:
#         f.write("Name\tR.T...s.\tArea\tQuant.Masses\tSpectra\n")
#         for identification_data_dict in matches_identification:
#             compound_name = identification_data_dict['compound_name']
#             rt1 = identification_data_dict['rt1']
#             rt2 = identification_data_dict['rt2']
#             area = identification_data_dict['area']
#             # formatted_spectrum = identification_data_dict['spectra']
#             formatted_spectrum = identification_data_dict.get('spectra', '')

#             f.write(write_line(compound_name, rt1, rt2, area,
#                                formatted_spectrum))


# def cohort_identification_alignment_input_format_txt(
#         filename, matches_identification, PATH):
#     with open(PATH + filename + '.txt', 'w', encoding='UTF8') as f:
#         f.write("Name\tR.T...s.\tArea\tQuant.Masses\tSpectra\n")
        
#         for d in matches_identification:
#             line = (
#                 f"{d['compound_name']}\t"
#                 f"{d['rt1']:.2f}\t"
#                 f"{d['rt2']:.2f}\t"
#                 f"{d['area']:.1f}\t"
#                 f"{d.get('spectra', '')}\n"
#             )
#             f.write(line)

def cohort_identification_alignment_input_format_txt(
        filename, matches_identification, PATH, deconvo=False):
    r"""Generate formatted peak table for alignment.

    Parameters
    ----------
    filename :
        Chromatogram full filename.
    matches_identification :
        Array of match dictionary containing casno, name, formula, spectra,
        coordinates...
    PATH : optional
        Path to the resulting formatted peak table.
    """
    if deconvo:
        deconvo_dir = os.path.join(PATH, 'deconvolution')
        os.makedirs(deconvo_dir, exist_ok=True)
        name_file = os.path.join(deconvo_dir, filename + '#Dc.txt')
        print("ðŸ“‚ /deconvolution directory created")
    else:
        name_file = PATH + filename + '.txt'

    with open(name_file, 'w', encoding='UTF8') as f:
        f.write("Name\tR.T...s.\tArea\tQuant.Masses\tSpectra\n")
        for identification_data_dict in matches_identification:
            compound_name = identification_data_dict['compound_name']
            rt1 = identification_data_dict['rt1']
            rt2 = identification_data_dict['rt2']
            area = identification_data_dict['area']
            if deconvo:
                formatted_spectrum = identification_data_dict['spectra_deconvo']
            else:
                formatted_spectrum = identification_data_dict['spectra']
            f.write(write_line(compound_name, rt1, rt2, area,
                               formatted_spectrum))


def sample_identification(path, file, output_path,
                          mod_time,
                          method="DoG", mode="mass_per_mass",
                          noise_factor=5, abs_thresholds=1000,
                          rel_thresholds=0.001,
                          cluster=0.5,
                          min_distance=1, min_sigma=1, max_sigma=3,
                          sigma_ratio=1.5,
                          num_sigma=10,
                          formated_spectra=True, match_factor_min=600,
                          min_persistence=0.0002,
                          overlap=0.5, eps=0.001, min_samples=1, nist=False,
                          method_baseline="als",
                          quant_method="mass", extract_patch=False,
                          output_hdf5_file=None, plot_=True):
    r"""Read sample chromatogram and generate the associated peak table.
    - identification()

    Parameters
    ----------
    path : str
        Path to the directory containing the chromatogram file.
    file : str
        Name of the chromatogram file.
    OUTPUT_PATH : str, optional
        Directory where the resulting peak table files will be saved. If None,
        results are saved in the current working directory.
    mod_time : float, default=1.25
        Modulation time used for chromatogram processing.
    method : str, default='persistent_homology'
        Method used for peak detection.
    mode : str, default='tic'
        Mode of chromatogram analysis.
    filtering_factor : int, default=1
        Detection threshold for peaks.
    hit_prob_min : int, default=15
        Minimum hit probability for compound identification.
    ABS_THRESHOLDS : list or None, default=None
        Absolute intensity thresholds for peak detection.
    cluster : bool, default=True
        Whether to apply clustering to detected peaks.
    min_distance : int, default=1
        Minimum distance between detected peaks.
    sigma_ratio : float, default=1.6
        Ratio used for Gaussian peak fitting.
    num_sigma : int, default=10
        Number of standard deviations to consider for peak detection.
    formated_spectra : bool, default=True
        Whether to format spectra before identification.
    match_factor_min : int, default=700
        Minimum match factor for compound identification.
    min_persistence : flaot
        Minimum persistence for peak detection with method persistent_homology.

    Examples
    --------
    >>> sample_identification("/path/to/data/", "sample.cdf")
    >>> # or with an output directory
    >>> sample_identification("/path/to/data/", "sample.cdf",
        OUTPUT_PATH="/path/to/results/")
    """

    #if os.path.exists(output_hdf5_file):
    #    raise FileExistsError(f"The file '{output_hdf5_file}' already exists.")
    if output_hdf5_file is None:
        output_hdf5_file = os.path.join(output_path, "data_set.h5")

    print('Identification started\n')
    start_time = time.time()
    try:
        full_filename = path + "/" + file
        result = identification(
            full_filename,
            output_path,
            mod_time,
            method,
            mode,
            noise_factor,
            abs_thresholds,
            rel_thresholds,
            cluster,
            min_distance,
            min_sigma,
            max_sigma,
            sigma_ratio,
            num_sigma,
            formated_spectra,
            match_factor_min,
            min_persistence,
            overlap,
            eps,
            min_samples,
            nist,
            quant_method,
            extract_patch,
            output_hdf5_file,
            method_baseline,
            plot_
        )
        # VÃ©rifier le type de rÃ©sultat
        if isinstance(result, tuple) and len(result) == 2:
            first, second = result
            if second == "persistent_homology_mode":
                coordinates = first
                return f"âœ… Peak detection completed: {len(coordinates)} peaks found"
            else:
                # Cas normal
                matches_identification, sample_metadata_list = result

                 # VÃ©rifier si les rÃ©sultats sont vides
                if not matches_identification and not sample_metadata_list:
                    return f"âŒ Aucune identification possible avec method='{method}' et mode='{mode}'"
                if not matches_identification:
                    return f"âš ï¸ Aucun composÃ© identifiÃ© pour {file}"

                print("Identification done", time.time()-start_time, 's')

                base_name = os.path.splitext(file)[0] + ('#Dt#N' if nist else '#Dt')
                cohort_identification_alignment_input_format_txt(
                    base_name, matches_identification, output_path)
                cohort_identification_alignment_input_format_txt(
                    base_name, matches_identification, output_path, deconvo=True)
                if (extract_patch):
                    cohort_identification_sample_metadata(
                        base_name, sample_metadata_list, output_path)
                else:
                    cohort_identification_to_csv(
                        base_name, matches_identification, output_path)
                result = [f'{output_path + base_name}.txt, {output_path + "deconvolution/" + base_name}#Dc.txt, {output_path + base_name}.csv created']
                return result
        else:
            return "âŒ Erreur inattendue lors de l'identification/peak detection."

    except Exception as e:
        traceback.print_exc()
        return (f"Erreur lors du traitement du fichier {file}: {e}")


def cohort_identification_sample_metadata(
        file, sample_metadata_list, output_path):
    file_path = output_path + '/' + file + "_sample_metadata.csv"

    fieldnames = sample_metadata_list[0].keys()
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(sample_metadata_list)


def clip_patch_by_rt(full_patch, full_rt_bounds, center_rt, clip_rt1_window, clip_rt2_window):
    """Clips the full_patch using RT mapping."""
    # (Implementation from previous versions)
    if full_patch.size == 0: return np.array([[]])
    (full_rt1_min, full_rt2_min), (full_rt1_max, full_rt2_max) = full_rt_bounds
    patch_height, patch_width = full_patch.shape
    desired_rt1_min = center_rt[0] - clip_rt1_window; desired_rt1_max = center_rt[0] + clip_rt1_window
    desired_rt2_min = center_rt[1] - clip_rt2_window; desired_rt2_max = center_rt[1] + clip_rt2_window
    target_rt1_min = max(desired_rt1_min, full_rt1_min); target_rt1_max = min(desired_rt1_max, full_rt1_max)
    target_rt2_min = max(desired_rt2_min, full_rt2_min); target_rt2_max = min(desired_rt2_max, full_rt2_max)
    if target_rt1_min >= target_rt1_max or target_rt2_min >= target_rt2_max: return np.array([[]])
    rt1_range_full = full_rt1_max - full_rt1_min; rt2_range_full = full_rt2_max - full_rt2_min
    if rt1_range_full <= 0 or rt2_range_full <= 0 or patch_height <= 1 or patch_width <= 1: return np.array([[]])
    x_clip_min_float = ((target_rt1_min - full_rt1_min) / rt1_range_full) * (patch_height -1)
    x_clip_max_float = ((target_rt1_max - full_rt1_min) / rt1_range_full) * (patch_height -1)
    y_clip_min_float = ((target_rt2_min - full_rt2_min) / rt2_range_full) * (patch_width -1)
    y_clip_max_float = ((target_rt2_max - full_rt2_min) / rt2_range_full) * (patch_width -1)
    x_clip_min_idx = int(round(x_clip_min_float)); x_clip_max_idx = int(round(x_clip_max_float)) + 1
    y_clip_min_idx = int(round(y_clip_min_float)); y_clip_max_idx = int(round(y_clip_max_float)) + 1
    x_clip_min_idx = max(0, x_clip_min_idx); y_clip_min_idx=max(0, y_clip_min_idx)
    x_clip_max_idx = min(patch_height, x_clip_max_idx); y_clip_max_idx=min(patch_width, y_clip_max_idx)
    if x_clip_min_idx >= x_clip_max_idx or y_clip_min_idx >= y_clip_max_idx: return np.array([[]])
    return full_patch[x_clip_min_idx:x_clip_max_idx, y_clip_min_idx:y_clip_max_idx]
