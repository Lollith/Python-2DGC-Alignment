import pandas as pd
import logging
import pyms
import pyms_nist_search

match_factor_min = 650

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

def matching_nist():
  #   search = pyms_nist_search.Engine(
#                     "C:/Users/camil/Documents/NIST/mainlib/",
#                     pyms_nist_search.NISTMS_MAIN_LIB,
#                     "C:/Users/camil/Documents/Python-2DGC",
#       
    search = pyms_nist_search.Engine(
        r"D:\Dossiers Persos\Adeline\app\nist\mainlib\\",
        pyms_nist_search.NISTMS_MAIN_LIB,
        r"D:\Dossiers Persos\Adeline\app\nist\tmp\\",
        )

    logger=logging.getLogger('pyms_nist_search')
    logger.setLevel('ERROR')
    logger=logging.getLogger('pyms')
    logger.setLevel('ERROR')

    # df = pd.read_csv("C:/Users/camil/data/td-ptr/gcxgc/resultPersistantHomology_tic/Align_table_info.csv",sep=";",  )
    df = pd.read_csv("annotate/Align_table_info.csv", sep=";")
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
        
        list_hits = []
        mass_spectrum = pyms.Spectrum.MassSpectrum(masses, intensities)

        hits = search.full_search_with_ref_data(mass_spectrum, n_hits=20)
        for i, hit_tuple in enumerate(hits):
            results = serialize_hit_tuple(hit_tuple)
            list_hits.append(results)
        top_hits = filter_best_hits(list_hits, match_factor_min)

        def join_field(field):
            return '/'.join(str(m.get(field, '')) for m in top_hits)

        if top_hits:
            identification_data_dict = {
            'compound_name': join_field('name'),
            'casno': join_field('cas_number'),
            'compound_formula': join_field('formula'),
            'match_factor': join_field('match_factor'),
        }
        else:
            identification_data_dict = {
            'compound_name': f"Analyte_{row+1}",
            'casno': '',
            'compound_formula': '',
            'match_factor': '',
            }
   
        for key in identification_data_dict:
            df.at[row, key] = identification_data_dict[key]

    # df.to_csv("C:/Users/camil/data/td-ptr/gcxgc/resultPersistantHomology_tic/Align_table_info_annotated.csv", index=False,sep=";")
    df.to_csv("Annotate/align_table_info_annotated_hits.csv", sep=";", index=False, encoding="utf-8-sig") #compatibilite avec excel


if __name__ == "__main__":
    matching_nist()