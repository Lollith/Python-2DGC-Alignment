
def serialize_hit_tuple(hit_tuple):
        search_result, ref_data = hit_tuple
        return {
            "name": getattr(search_result, "name", None),
            "match_factor": getattr(search_result, "match_factor", None),
            "cas_number": getattr(search_result, "cas", None),
            "formula": getattr(ref_data, "formula", None),
        }

def filter_best_hits(list_hits, match_factor_min):
    match_factors = [hit["match_factor"] for hit in list_hits]
    max_match_factor = max(match_factors, default=0)

    filtered_hits = [
        hit for hit in list_hits
        if hit["match_factor"] >= max_match_factor - 100
        and hit["match_factor"] >= match_factor_min
    ]
    return filtered_hits
import nist_search

def matching_nist_lib_from_chromato_cube():
    # print("matching nist, chromatocube shape", chromato_cube.shape)


    matches = []
    nb_analyte = 0
    # top_hits = []
    # serialized_spectra = []
    nist_api = nist_search.NISTSearchWrapper() 

        # mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        spectrum_hash = hash_spectrum(mass_values, int_values)

        # serialized_spectrum = {
        #     "mass": [float(m) for m in mass_values],
        #     "intensity": [float(i) for i in int_values]
        #     }

        if nist_api.check_nist_health():
                # print("Matching with NIST library...")
                serialized_spectrum = {
                    "mass": [float(m) for m in mass_values],
                    "intensity": [float(i) for i in int_values]
                }
                results = nist_api.nist_single_search(serialized_spectrum)
                list_hit = nist_api.hit_list_from_nist_api(results)
                top_hits = filter_best_hits(list_hit, match_factor_min)
        else:
            print(f"[Peak {i + 1}] NIST API unavailable or skipped.")
            top_hits = []

        match_results = []
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
                    'reverse_match_factor': search_result.reverse_match_factor,
                    'spectra': int_values
                    }
                match_results.append(match_data)
        # else:
        #     # Composé non identifié
        #     nb_analyte += 1
        #     match_results.append({
        #         'spectra': int_values,
        #         'compound_name': f'Analyte{nb_analyte}',
        #         'casno': '',
        #         'compound_formula': '',
        #         'hit_prob': '',
        #         'match_factor': '',
        #         'reverse_match_factor': ''
        #         })

        matches.append([[(coordinates_in_chromato[i][0]),
                       (coordinates_in_chromato[i][1])], match_results, coord])
    end = time.time() - start
    print(f"Matching NIST library took {end:.2f} seconds")

    return matches