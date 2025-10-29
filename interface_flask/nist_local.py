import pyms_nist_search
import os
import pyms.Spectrum
from dotenv import load_dotenv

load_dotenv()

def search(input_path, output_path, files):
    """
    Perform a local NIST search operation.

    :param input_path: Path to the input data.
    :param output_path: Path to save the output results.
    :param files: List of files to process.
    :return: Search results.
    """
    print("Starting NIST local search...")
    # mainlib_path = os.getenv("MAINLIB_PATH", "C:/NIST20/MSSEARCH/mainlib")
    # temp_dir = os.getenv("TEMP_DIR", "C:/NIST20/MSSEARCH/tmp")

    # engine = pyms_nist_search.Engine(
    #                mainlib_path,
    #                 pyms_nist_search.NISTMS_MAIN_LIB,
    #                 temp_dir
    #                 )
    # match = []
    # mass_values = []
    # int_values = []
    # for file in files:
    #     filepath = os.path.join(input_path, file)
    #     print(f"Processing file: {filepath}")
    #     # lire le file.csv et chercher la colonne  spectra  et extraire mass_values et int_values tel que masse:intensite masse:intensite ...
    #     with open(filepath, 'r') as f:
    #         for line in f:
    #             if line.startswith("spectra"):
    #                 # extraire les valeurs de masse et d'intensité
    #                 parts = line.strip().split(" ")
    #                 for part in parts[1:]:
    #                     mass, intensity = part.split(":")
    #                     mass_values.append(float(mass))
    #                     int_values.append(float(intensity))
    #                     print(f"mass: {mass}, intensity: {intensity}")
    #     # faire la recherche NIST

        # sortie creer un file ds l output-path
    result = True
    return result
    


    #     mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
    #     res = search.full_search_with_ref_data(mass_spectrum)
    #     #res = search.full_spectrum_search(mass_spectrum)
    #     if (res[0][0].match_factor < match_factor_min):
    #         continue
    #     '''if (res[0][0].hit_prob < hit_prob_min):
    #         continue'''
    #     #print(res[0][1].formula)
    #     del mass_spectrum
    #     compound_casno = res[0][0].cas
    #     compound_name = res[0][0].name
    #     compound_formula = res[0][1].formula
    #     hit_prob = res[0][0].hit_prob
    #     match_factor = res[0][0].match_factor
    #     reverse_match_factor = res[0][0].reverse_match_factor
    #     d_tmp['casno'] = compound_casno
    #     d_tmp['compound_name'] = compound_name
    #     d_tmp['compound_formula'] = compound_formula
    #     d_tmp['hit_prob'] = hit_prob
    #     d_tmp['match_factor'] = match_factor
    #     d_tmp['reverse_match_factor'] = reverse_match_factor
        
    #     d_tmp['spectra'] = int_values
    #     '''if (res[0][0].hit_prob < hit_prob_min):
    #         nb_analyte = nb_analyte + 1
    #         d_tmp['compound_name'] = 'Analyte' + str(nb_analyte)'''
        
    #     match.append([[(coordinates_in_chromato[i][0]), (coordinates_in_chromato[i][1])], d_tmp, coord])
        
    #     del res
    # print("nb match:")
    # print(len(coordinates))
    # return match


