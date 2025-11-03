import pyms_nist_search
import os
import pyms.Spectrum
from dotenv import load_dotenv
import time
import csv
import pandas as pd

load_dotenv()

# TODO fonction de matching a importer
# def filter_best_hits(list_hits, match_factor_min):
#     search_results = [hit[0] for hit in list_hits]
#     match_factors = [result.match_factor for result in search_results]
#     max_match_factor = max(match_factors)

#     filtered_hits = [
#         result for result in search_results
#         if result.match_factor >= max_match_factor - 100
#         and result.match_factor >= match_factor_min
#     ]
#     return filtered_hits

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

# def save_results_to_csv_native(input_path, output_path, files_list, all_matches, messages):
#     """Sauvegarder avec le module CSV natif Python"""
    
#     for file in files_list:
#         input_filepath = os.path.join(input_path, file)
#         output_filepath = os.path.join(output_path, f"identified_{file}")
        
#         # ✅ Lire le fichier original
#         with open(input_filepath, 'r', encoding='utf-8') as infile:
#             reader = csv.reader(infile, delimiter=';')
#             data = list(reader)
        
#         if not data:
#             continue
            
#         # ✅ Ajouter les nouveaux en-têtes
#         header = data[0]
#         new_columns = ['NIST_Compound_Name', 'NIST_CAS', 'NIST_Formula', 
#                       'NIST_Match_Factor', 'NIST_Hit_Prob', 'NIST_Reverse_Match']
#         header.extend(new_columns)
        
#         # ✅ Initialiser les nouvelles colonnes vides pour chaque ligne
#         for i in range(1, len(data)):
#             data[i].extend([''] * len(new_columns))
        
#         # ✅ Remplir avec les résultats d'identification
#         for match in all_matches:
#             if match['file'] == file:
#                 line_idx = match['line']  # Line 1-based
#                 if line_idx < len(data):
#                     row = data[line_idx]
#                     # Index des nouvelles colonnes
#                     base_idx = len(row) - len(new_columns)
#                     row[base_idx] = str(match['compound_name'])
#                     row[base_idx + 1] = str(match['casno'])
#                     # row[base_idx + 2] = str(match['compound_formula'])
#                     row[base_idx + 3] = str(match['match_factor'])
#                     row[base_idx + 4] = str(match['hit_prob'])
#                     row[base_idx + 5] = str(match['reverse_match_factor'])
        
#         # ✅ Écrire le fichier de sortie
#         with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
#             writer = csv.writer(outfile, delimiter=';')
#             writer.writerows(data)
        
#         messages.append(f"✅ Fichier CSV identifié sauvegardé: {output_filepath}")

    

def matching_nist(input_path, output_path, files):
    """
    Perform a local NIST search operation.

    :param input_path: Path to the input data.
    :param output_path: Path to save the output results.
    :param files: List of files to process.
    :return: Search results.
    """

    messages = []

    messages.append("🔬 Starting NIST local search...")
    start = time.time()
    
    mainlib_path = os.getenv("MAINLIB_PATH", "C:/NIST20/MSSEARCH/mainlib")
    temp_dir = os.getenv("TEMP_DIR", "C:/NIST20/MSSEARCH/tmp")

    files_list = [f.strip() for f in files.split(',') if f.strip()]
    messages.append(f"📋 Fichiers à traiter: {files_list}")

    with pyms_nist_search.Engine(mainlib_path, pyms_nist_search.NISTMS_MAIN_LIB, temp_dir) as engine:
    # engine = pyms_nist_search.Engine(
    #                mainlib_path,
    #                 pyms_nist_search.NISTMS_MAIN_LIB,
    #                 temp_dir
    #                 )
        # all_matches = []
        # mass_values = []
        # int_values = []
        
        for file in files_list:
            filepath = os.path.join(input_path, file)
            messages.append(f"Processing file: {filepath}")
        #     # lire le file.csv et chercher la colonne  spectra  et extraire mass_values et int_values tel que masse:intensite masse:intensite ...
            df = pd.read_csv(filepath, sep=";")
            df['compound_name'] = ""
            df['casno'] = ""
            df['compound_formula'] = ""
            df['match_factor'] = ""
            
            compounds_processed = 0
            compounds_identified = 0

            for row in range(df.shape[0]):
                # if row % 100 == 0:  # Tous les 100 composés
                #     messages.append(f"📊 Traité {row}/{len(df)} composés...")

                s = df.at[row, 'Spectra']
                pairs = s.strip().split()

                masses = []
                intensities = []

                for pair in pairs:
                        m, i = pair.split(":")
                        masses.append(int(float(m)))  #  or float(m) if decimals matter
                        intensities.append(float(i))

                compounds_processed += 1

                list_hits = []
                mass_spectrum = pyms.Spectrum.MassSpectrum(masses, intensities)

                hits = engine.full_search_with_ref_data(mass_spectrum, n_hits=20)
                for i, hit_tuple in enumerate(hits):
                    results = serialize_hit_tuple(hit_tuple)
                    list_hits.append(results)
                match_factor_min = 650 #TODO parametre ??
                top_hits = filter_best_hits(list_hits, match_factor_min)

                def join_field(field):
                    return '/'.join(str(m.get(field, '')) for m in top_hits)

                if top_hits:
                    coumpounds_identified += 1
                    identification_data_dict = {
                    'compound_name': join_field('name'),
                    'casno': join_field('cas_number'),
                    'compound_formula': join_field('formula'),
                    'match_factor': join_field('match_factor'),
                }
                    for key in identification_data_dict:
                        df.at[row, key] = identification_data_dict[key]

            output_filepath = os.path.join(output_path, f"identified2_{file}")
            df.to_csv(output_filepath, sep=";", index=False, encoding="utf-8-sig") #compatibilite avec excel
            messages.append(f"✅ {file}: {compounds_identified}/{compounds_processed} composés identifiés")
            messages.append(f"💾 Fichier CSV identifié sauvegardé: {output_filepath}")
        # all_matches = []  # Placeholder to return if needed
                
            # with open(filepath, 'r') as f:
            #     first_line = f.readline().strip()
            #     # messages.append(f"First line of the file: {first_line}")
                
            #     if "Spectra" in first_line:
            #         # messages.append(f"Found spectra line: {first_line.strip()}")
            #         headers = first_line.split(';')  # Séparer par point-virgule
            #         # messages.append(f"📊 Colonnes trouvées: {headers}")
                        
            #         spectra_index = headers.index("Spectra")
            #         # messages.append(f"📍 Colonne 'Spectra' à l'index: {spectra_index}")

            #         line_count = 0
            #         nb_analyte = 0
            #         for line in f:
            #             line_count += 1
            #             parts = line.strip().split(';')

            #             if len(parts) > spectra_index: 
            #                 mass_values = []
            #                 int_values = []
            #                 spectra_data = parts[spectra_index]
            #                 # messages.append(f"Spectra data: {spectra_data}")
                            
            #                 spectra_parts = spectra_data.split(" ")
            #                 for part in spectra_parts:
            #                     if ':' in part:
            #                         mass, intensity = part.split(":")
            #                         mass_values.append(float(mass))
            #                         int_values.append(float(intensity))
            #                 # messages.append(f"Extracted mass values: {mass_values}")
            #                 # messages.append(f"Extracted intensity values: {int_values}")

            #             mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
            #             results = engine.full_search_with_ref_data(mass_spectrum)
            #             match_factor_min = 650 # TODO parametre ??
            #             top_hits = filter_best_hits(results, match_factor_min)
                    
            #             if top_hits:
            #                 for j, hit in enumerate(top_hits):
            #                     match_data = {
            #                         'file': file,
            #                         'line': line_count,
            #                         'number': j,
            #                         'casno': hit.cas,
            #                         'compound_name': hit.name,
            #                         # 'compound_formula': hit.formula,
            #                         'hit_prob': hit.hit_prob,
            #                         'match_factor': hit.match_factor,
            #                         'reverse_match_factor': hit.reverse_match_factor,
            #                         'spectra': int_values
            #                         }
            #                     all_matches.append(match_data)
            #             else:
            #                 # Composé non identifié
            #                 nb_analyte += 1
            #                 all_matches.append({
            #                     'file': file,
            #                     'line': line_count,
            #                     'spectra': int_values,
            #                     'compound_name': f'Analyte{nb_analyte}',
            #                     'casno': '',
            #                     'compound_formula': '',
            #                     'hit_prob': '',
            #                     'match_factor': '',
            #                     'reverse_match_factor': ''
            #                     })



        # sortie creer un file ds l output-path
    # messages.append("🔓 Moteur NIST libéré automatiquement")
    # try:
    #     save_results_to_csv_native(input_path, output_path, files_list, all_matches, messages)
    # except Exception as e:
    #     messages.append(f"❌ Erreur sauvegarde: {str(e)}")
    end = time.time() - start
    messages.append("⏱️ NIST local search completed in {end:.2f} secondes")

    return messages



