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
from pyms_nist_search.search_result import SearchResult

# present = {"HMDB0031018": [[23.43, 0.008]], "HMDB0061859": [[32.22, 0.042]], "HMDB0030469": [[28.15, 0.008]],
#            "HMDB0031264": [[15.59, 0.017]], "HMDB0033848": [[18.41, 0.025]], "HMDB0031291": [[13.10, 1.231]],
#            "HMDB0034154": [[36.08, 0.083]]}

# present = ['111-82-0', '112-39-0', '124-10-7', '1731-84-6', '110-42-9', '111-11-5', '112-61-8', '1120-28-1', '929-77-1', '5802-82-4', '55682-92-3', '2442-49-1']

# def check_match(match):
#     #return np.array([databaseid for databaseid in [meta['databaseid'] for meta in match[:, 1]] if databaseid in present])
#     return np.array([databaseid for databaseid in [meta['casno'] for meta in match[:, 1]] if databaseid in present])

# '''def check_match_nist_lib(match):
#     return np.array([casno for casno in match[:, 1] if casno in present])'''


def hit_list_with_ref_data_from_json(json_data: str) \
    -> List[Tuple[SearchResult, ReferenceData]]:
    """
    Parse json data into a list of (SearchResult, ReferenceData) tuples.
    :param json_data: str
    """

    raw_output = json.loads(json_data)

    hit_list = []

    for hit, ref_data in raw_output:
        hit_list.append((SearchResult(**hit), ReferenceData(**ref_data)))
    return hit_list

def full_search_with_ref_data(
            mass_spectrum,
            n_hits: int = 20,
            ) -> List[Tuple[SearchResult, ReferenceData]]:
        """
        Perform a Full Spectrum Search of the mass spectral library, including reference data.

        :param mass_spec: The mass spectrum to search against the library.
        :param n_hits: The number of hits to return.

        :return: List of tuples containing possible identities
            for the mass spectrum, and the reference data.
        """

        if not isinstance(mass_spectrum, pyms.Spectrum.MassSpectrum):
            raise TypeError("`mass_spec` must be a pyms.Spectrum.MassSpectrum object.")

        retry_count = 0

        # Keep trying until it works
        while retry_count < 240:
            try:
                res = requests.post(
                        f"http://nist:5001/search/spectrum_with_ref_data/?n_hits={n_hits}",
                        json=sdjson.dumps(mass_spectrum)
                        )
                return hit_list_with_ref_data_from_json(res.text)
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
                retry_count += 1

        raise TimeoutError("Unable to communicate with the search server.")


def filter_best_hits(list_hits, match_factor_min):
    match_factors = [hit[0].match_factor for hit in list_hits]
    max_match_factor = max(match_factors)

    filtered_hits = [
        hit for hit in list_hits
        if hit[0].match_factor >= max_match_factor - 100
        and hit[0].match_factor >= match_factor_min
    ]
    return filtered_hits


def matching_nist_lib_from_chromato_cube(
        chromato_obj, chromato_cube, coordinates, mod_time,
        match_factor_min, nist):
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
        (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    except ValueError:
        range_min, range_max = spectra_obj

    mass_values = np.linspace(
        range_min, range_max, range_max - range_min + 1).astype(int)

    matches = []
    nb_analyte = 0
    for i, coord in enumerate(coordinates):
        int_values = mass_spec.read_spectrum_from_chromato_cube(
            coord, chromato_cube=chromato_cube)
        mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        match_results = []

        if nist: # TODO check ici
            print("Matching with NIST library...")
            list_hit = full_search_with_ref_data(mass_spectrum, n_hits=20)
            top_hits = filter_best_hits(list_hit, match_factor_min)
            print(f"peak {i + 1} has {len(top_hits)} hits for {coord} and with match_factor_min={match_factor_min}.")
        
        if top_hits:
            for j, hit in enumerate(top_hits):
                search_result, ref_data = hit
                print(f"hit {j}: {search_result.name}: {search_result.cas}, "
                      f"with match_factor:{search_result.match_factor}.")

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
            # Composé non identifié
            nb_analyte += 1
            match_results.append({
                'spectra': int_values,
                'compound_name': f'Analyte{nb_analyte}',
                'casno': '',
                'compound_formula': '',
                'hit_prob': '',
                'match_factor': '',
                'reverse_match_factor': ''
                })
        print()

        matches.append([[(coordinates_in_chromato[i][0]),
                       (coordinates_in_chromato[i][1])], match_results, coord])
        del mass_spectrum
    end = time.time() - start
    print(f"Matching NIST library took {end:.2f} seconds")

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
