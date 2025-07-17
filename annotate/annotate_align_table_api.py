import pandas as pd
import logging
import pyms
import nist_search 

nist_api_search = nist_search.NISTSearchWrapper()
match_factor_min = 650

def filter_best_hits(list_hits, match_factor_min):
    match_factors = [hit[0].match_factor for hit in list_hits]
    max_match_factor = max(match_factors)

    filtered_hits = [
        hit for hit in list_hits
        if hit[0].match_factor >= max_match_factor - 100
        and hit[0].match_factor >= match_factor_min
    ]
    return filtered_hits

def extract_hits_for_csv(top_hits):
    """
    Transforme une liste de hits [(search_result, ref_data), ...] 
    en dictionnaire avec les champs concaténés par "/"
    """
    def join(field):
        return '/'.join(
            str(getattr(hit[0], field, '')) for hit in top_hits
        )

    def join_ref(field):
        return '/'.join(
            str(getattr(hit[1], field, '')) for hit in top_hits
        )

    return {
        'compound_name': join('name'),
        'casno': join('cas'),
        'compound_formula': join_ref('formula'),
        'match_factor': join('match_factor')
    }


df = pd.read_csv("Align_table_info.csv", sep=";")
df['compound_name'] = ""
df['casno'] = ""
df['compound_formula'] = ""
df['match_factor'] = ""

for row in range(df.shape[0]):
    s = df.at[row, 'Spectra']
    pairs = s.strip().split()

    masses = []
    intensities = []

    for pair in pairs:
            m, i = pair.split(":")
            masses.append(int(float(m)))  #  or float(m) if decimals matter
            intensities.append(float(i))
    
    serialized_spectrum = {
    "masses": masses,
    "intensities": intensities
    }
    print (f"Serialized spectrum for row {row}: {serialized_spectrum}")

    # spectrum = pyms.Spectrum.MassSpectrum(masses, intensities)

    try:
        if nist_api_search.check_nist_health():
            results = nist_api_search.nist_single_search(serialized_spectrum)
            list_hit = nist_api_search.hit_list_from_nist_api(results)
            top_hits = filter_best_hits(list_hit, match_factor_min)
            print(f"Peak {i + 1} has {len(top_hits)} hits for {coord}.")

            if top_hits:
                fields = extract_hits_for_csv(top_hits)
                df.at[row, 'compound_name'] = fields['compound_name']
                df.at[row, 'casno'] = fields['casno']
                df.at[row, 'compound_formula'] = fields['compound_formula']
                df.at[row, 'match_factor'] = fields['match_factor']

    except Exception as e:
        print(f"Erreur à la ligne {row} : {e}")

df.to_csv("Align_table_info_with_nist.csv", sep=";", index=False, encoding="utf-8-sig") #compatibilite avec excel