import numpy as np
import projection
import mass_spec
import pyms
import requests
import time
import sdjson
import json
from typing import List, Tuple
from nist_utils.reference_data import ReferenceData
#from pyms_nist_search.search_result import SearchResult
import nist_search

# present = {"HMDB0031018": [[23.43, 0.008]], "HMDB0061859": [[32.22, 0.042]], "HMDB0030469": [[28.15, 0.008]],
#            "HMDB0031264": [[15.59, 0.017]], "HMDB0033848": [[18.41, 0.025]], "HMDB0031291": [[13.10, 1.231]],
#            "HMDB0034154": [[36.08, 0.083]]}

# present = ['111-82-0', '112-39-0', '124-10-7', '1731-84-6', '110-42-9', '111-11-5', '112-61-8', '1120-28-1', '929-77-1', '5802-82-4', '55682-92-3', '2442-49-1']

# def check_match(match):
#     #return np.array([databaseid for databaseid in [meta['databaseid'] for meta in match[:, 1]] if databaseid in present])
#     return np.array([databaseid for databaseid in [meta['casno'] for meta in match[:, 1]] if databaseid in present])

# '''def check_match_nist_lib(match):
#     return np.array([casno for casno in match[:, 1] if casno in present])'''


# def hit_list_with_ref_data_from_json(json_data: str) \
#     -> List[Tuple[SearchResult, ReferenceData]]:
#     """
#     Parse json data into a list of (SearchResult, ReferenceData) tuples.
#     :param json_data: str
#     """

#     raw_output = json.loads(json_data)

#     hit_list = []

#     for hit, ref_data in raw_output:
#         hit_list.append((SearchResult(**hit), ReferenceData(**ref_data)))
#     return hit_list

# def full_search_with_ref_data(
#             mass_spectrum,
#             n_hits: int = 20,
#             ) -> List[Tuple[SearchResult, ReferenceData]]:
#         """
#         Perform a Full Spectrum Search of the mass spectral library, including reference data, using local Flask API.

#         :param mass_spec: The mass spectrum to search against the library.
#         :param n_hits: The number of hits to return.

#         :return: List of tuples (SearchResult, ReferenceData)
#         """

#         if not isinstance(mass_spectrum, pyms.Spectrum.MassSpectrum):
#             raise TypeError("`mass_spec` must be a pyms.Spectrum.MassSpectrum object.")

#         retry_count = 0

#         # Keep trying until it works
#         while retry_count < 240:
#             try:
#                 res = requests.post(
#                         f"http://nist:5001/search/spectrum_with_ref_data/?n_hits={n_hits}",
#                         json=sdjson.dumps(mass_spectrum)
#                         )
#                 return hit_list_with_ref_data_from_json(res.text)
#             except requests.exceptions.ConnectionError:
#                 time.sleep(0.5)
#                 retry_count += 1

#         raise TimeoutError("Unable to communicate with the search server.")
# def hit_list_from_nist_api(result_json):
#     """
#     Transforme le résultat JSON de l'API Flask NIST en liste de tuples (SearchResult, ReferenceData)
#     """
#     hit_list = []
#     for hit in result_json.get("hits", []):
#         # Adapte selon tes classes SearchResult et ReferenceData
#         search_result = SearchResult(
#             name=hit.get("name"),
#             match_factor=hit.get("match_factor"),
#             reverse_match_factor=hit.get("reverse_match"),
#             hit_prob=None,  # Si tu n'as pas cette info
#             cas=hit.get("cas_number")
#         )
#         ref_data = ReferenceData(
#             formula=hit.get("formula"),
#             molecular_weight=hit.get("molecular_weight")
#         )
#         hit_list.append((search_result, ref_data))
#     return hit_list


# def nist_batch_search(list_mass_spectra, api_url="http://localhost:8080/nist/batch_search"):
#     """
#     Envoie une liste de spectres à l'API Flask NIST pour identification en lot.
#     """
#     spectra_data = [spec.to_dict() for spec in list_mass_spectra]  # Adapte si besoin
#     retry_count = 0
#     while retry_count < 10:
#         try:
#             res = requests.post(api_url, json={"spectra": spectra_data})
#             res.raise_for_status()
#             return res.json()["results"]
#         except requests.exceptions.ConnectionError:
#             time.sleep(0.5)
#             retry_count += 1
#     raise TimeoutError("Unable to communicate with the NIST API server.")


def filter_best_hits(list_hits, match_factor_min):
    match_factors = [hit[0].match_factor for hit in list_hits]
    max_match_factor = max(match_factors)

    filtered_hits = [
        hit for hit in list_hits
        if hit[0].match_factor >= max_match_factor - 100
        and hit[0].match_factor >= match_factor_min
    ]
    return filtered_hits

def search_and_filter(i, coord, spectrum, coordinates_in_chromato,
                      match_factor_min):
    try:
        print(f"[THREAD {i}] starting search...")
        nist_api = nist_search.NISTSearchWrapper()

        if not nist_api.check_nist_health():
            print("NIST API is not available. Skipping search.")
            return []
        
        print("NIST API is available. Proceeding with search.")
        result = nist_api.nist_single_search(spectrum)
        hits = nist_api.hit_list_from_nist_api(result)
        # print(f"[THREAD {i}] hits bruts = {[h[0].match_factor for h in hits]}")
        top_hits = filter_best_hits(hits, match_factor_min)
        print(f"[THREAD {i}] top_hits = {top_hits}", flush=True)
        match_results = []

        if top_hits:
            for j, hit in enumerate(top_hits):
                search_result, ref_data = hit
                match_data = {
                    'number': j,
                    'casno': search_result.cas,
                    'compound_name': search_result.name,
                    'compound_formula': ref_data.formula,
                    'hit_prob': search_result.hit_prob,
                    'match_factor': search_result.match_factor,
                    'reverse_match_factor': search_result.reverse_match_factor
                }
                match_results.append(match_data)
        else:
            match_results.append({
                'spectra': spectrum["intensity"],
                'compound_name': f'Analyte{i + 1}',
                'casno': '',
                'compound_formula': '',
                'hit_prob': '',
                'match_factor': '',
                'reverse_match_factor': ''
            })

        return [[coordinates_in_chromato[i][0], coordinates_in_chromato[i][1]], match_results, coord]
    except Exception as e:
        print(f"❌ Erreur dans search_and_filter pour le pic {i} : {e}")
        return [[coordinates_in_chromato[i][0], coordinates_in_chromato[i][1]], [], coord]


from concurrent.futures import ThreadPoolExecutor, as_completed

def matching_nist_lib_from_chromato_cube(chromato_obj, 
                                         chromato_cube, coordinates, mod_time,match_factor_min, nist=True):
    """Indentify retrieved peaks using NIST library.

    Parameters
    ----------
    chromato_obj :
        Chromatogram object wrapping chromato, time_rn and spectra_obj
    chromato_cube :
        3D chromatogram.
    coordinates :
        Peaks coordinates.
    mod_time : 
        Modulation time
    match_factor_min :
        Match factor between our spectrum and the library spectrum NIST.
        The match factor is a measure of how well the two spectra       
        match. It is calculated as the sum of the squares of the
        differences between the intensities of the two spectra.     
        A match factor of 1000 or higher is considered a good match.
        A match factor of 800 or higher is considered a fair match.
        A match factor of 600 or lower is considered a poor match.
        Filter compounds with match_factor <pmatch_factor_min
    -------
    Returns
    -------
    matches:
        Array of match dictionary containing casno, name, formula and spectra
        for each of identified as well as hit_prob, match_factor and
        reverse_match_factor
    --------
    """

    start = time.time()
    chromato, time_rn, spectra_obj = chromato_obj
    coordinates_in_chromato = projection.matrix_to_chromato(
        coordinates, time_rn, mod_time, chromato.shape)

    try:
        (_, _, _, _, range_min, range_max) = spectra_obj
    except ValueError:
        range_min, range_max = spectra_obj

    mass_values = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    serialized_spectra = [
        {
            "mass": [float(m) for m in mass_values],
            "intensity": [float(i) for i in mass_spec.read_spectrum_from_chromato_cube(coord, chromato_cube)]
        }
        for coord in coordinates
    ]

    if not nist:
        print("⚠️ NIST matching is disabled.")
        return [
            [[coordinates_in_chromato[i][0], coordinates_in_chromato[i][1]], [], coord]
            for i, coord in enumerate(coordinates)
        ]

    print("🔍 Starting NIST matching...")

    matches = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                search_and_filter,
                i, coord, spectrum, coordinates_in_chromato, match_factor_min
            ): i
            for i, (coord, spectrum) in enumerate(zip(coordinates, serialized_spectra))
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result(timeout=30)
                matches.append(result)
                print(f"✅ Peak {i + 1} has been processed.", flush=True)
            except Exception as e:
                print(f"❌ Error processing peak {i + 1}: {e}")
                matches.append([[coordinates_in_chromato[i][0], coordinates_in_chromato[i][1]], [], coordinates[i]])

    print(f"⏱️ Matching NIST library took {time.time() - start:.2f} seconds")
    return matches


# def mass_spectra_to_mgf(spectrum_path, mass_values_list, intensity_values_list):

# def matching_nist_lib(chromato_obj, spectra, coordinates, mod_time = 1.25):

#     chromato, time_rn, spectra_obj = chromato_obj
#     coordinates_in_chromato = projection.matrix_to_chromato(
#         coordinates, time_rn, mod_time, chromato.shape)
#     search = pyms_nist_search.Engine(
#                     "C:/NIST14/MSSEARCH/mainlib/",
#                     pyms_nist_search.NISTMS_MAIN_LIB,
#                     "C:/Users/Stan/Test",
#                     )
#     match = []
#     for i, coord in enumerate(coordinates):
        
#         d_tmp = dict()
#         mass_values, int_values = mass_spec.read_spectrum(chromato, coord, spectra)
#         mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
#         res = search.full_search_with_ref_data(mass_spectrum)
        
        
#         del mass_spectrum
        
        
#         '''if (res[0][0].hit_prob < 15):
#             continue'''
#         compound_casno = res[0][0].cas
#         compound_name = res[0][0].name
#         d_tmp['casno'] = compound_casno
#         d_tmp['compound_name'] = compound_name
#         match.append([[round(coordinates_in_chromato[i][0], 2), round(coordinates_in_chromato[i][1], 2)], d_tmp])
        
#         del res
#     return np.array(match)


# def matching(chromato_obj, spectra, spectrum_path, spectrum_lib, coordinates, mod_time=1.25, min_score=None):
#     chromato, time_rn, spectra_obj = chromato_obj
#     coordinates_in_chromato = projection.matrix_to_chromato(
#         coordinates, time_rn, mod_time, chromato.shape)

#     mass_values_list = []
#     intensity_values_list = []

#     for coord in coordinates:
#         mass_values, int_values = mass_spec.read_spectrum(chromato, coord, spectra)
#         mass_values_list.append(mass_values)
#         intensity_values_list.append(int_values)

#     mass_spectra_to_mgf(spectrum_path, mass_values_list, intensity_values_list)

#     spectrum = list(load_from_mgf(spectrum_path, metadata_harmonization=False))
#     matching = mass_spec.peak_retrieval(spectrum, spectrum_lib, coordinates_in_chromato, min_score=min_score)

#     return matching

# def compute_metrics_from_chromato_cube(coordinates, chromato_obj, chromato_cube, spectrum_path, spectrum_lib, mod_time=1.25):
#     if (not len(coordinates)):
#         return [0,0,0,0]
#     else:
#         match = matching_nist_lib_from_chromato_cube(chromato_obj=chromato_obj, chromato_cube=chromato_cube, coordinates=coordinates)
#         found_present = check_match(match)
#         #nb_peaks,rappel,precision
#         return [len(coordinates), len(np.unique(found_present)) / len(present), len(found_present) / len(coordinates), len(found_present)]
# import gc
# def compute_metrics(coordinates, chromato_obj, spectra, spectrum_path, spectrum_lib, mod_time=1.25):
#     if (not len(coordinates)):
#         return [0,0,0,0]
#     else:
#         '''match = matching(chromato_obj, spectra, spectrum_path,
#                      spectrum_lib, coordinates, mod_time=mod_time)'''
#         match = matching_nist_lib(chromato_obj=chromato_obj, spectra=spectra, coordinates=coordinates)
#         match_np_array=np.array(match)
#         found_present = check_match(match_np_array)
#         #nb_peaks,rappel,precision
#         return [len(coordinates), len(np.unique(found_present)) / len(present), len(found_present) / len(coordinates), len(found_present)]
