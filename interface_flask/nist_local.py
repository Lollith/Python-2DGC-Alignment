import pyms_nist_search
import os
import pyms.Spectrum
from dotenv import load_dotenv
import time
import pandas as pd
import json

load_dotenv()


class NistLocal:
    def __init__(self):
        self.mainlib_path = os.getenv("MAINLIB_PATH", "C:/NIST20/MSSEARCH/mainlib")
        self.temp_dir = os.getenv("TEMP_DIR", "C:/NIST20/MSSEARCH/tmp")

    def get_peak_info_files_from_folders(self, path):
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if "peak_info" in f.lower()]
            return files
        else:
            return []

    def serialize_hit_tuple(self, hit_tuple):
        search_result, ref_data = hit_tuple
        return {
            "name": getattr(search_result, "name", None),
            "match_factor": getattr(search_result, "match_factor", None),
            "cas_number": getattr(search_result, "cas", None),
            "formula": getattr(ref_data, "formula", None),
        }


    def filter_best_hits(self, list_hits, match_factor_min):
        match_factors = [hit["match_factor"] for hit in list_hits]
        max_match_factor = max(match_factors, default=0)
        filtered_hits = [
            hit for hit in list_hits
            if hit["match_factor"] >= max_match_factor - 100
            and hit["match_factor"] >= match_factor_min
        ]

        return filtered_hits

    def get_files_from_folder(self, path):
        """Get all peak_info files from a folder."""
        if os.path.isdir(path):
            return [f for f in os.listdir(path)
                    if 'peak_info' in f.lower()
                    and not f.startswith('identified_')]
        else:
            return []
        
    def check_files(self, input_path, files):
        messages = []

        if files is None:
            # si pas de file specifique, selectionne tous les peak info ds le dossier
            files_list = self.get_peak_info_files_from_folders(input_path)
            messages.append(f"❌ no selected filed, search file{files_list}")
            
        else:
            # fichier specifique fournit
            files_list = [f.strip() for f in files.split(',') if f.strip()]
            messages.append(f"❌ selected file :{files_list}")
        
        if not files_list:
            messages.append("❌ Aucun fichier Peak_Info trouvé !")
            return None, messages
        
        
        messages.append(f"🎯 Fichiers peak_info à traiter: {files_list}")
        return files_list, messages

    def matching_nist(self, input_path, output_path, files_list):
        yield "? Starting NIST local search..."
        start = time.time()
        
        with pyms_nist_search.Engine(
                self.mainlib_path,
                pyms_nist_search.NISTMS_MAIN_LIB,
                self.temp_dir) as engine:
            
            for file in files_list:
                filepath = os.path.join(input_path, file)
                yield f"? Processing file: {filepath}"  # ? Message immédiat !
                
                df = pd.read_csv(filepath, sep=";")
                df['compound_name'] = ""
                df['casno'] = ""
                df['compound_formula'] = ""
                df['match_factor'] = ""

                compounds_processed = 0
                compounds_identified = 0
                total_rows = df.shape[0]
                
                for row in range(total_rows):
                    # ? Messages de progression
                    if row % 20 == 0:
                        progress = (row / total_rows) * 100
                        yield f"? {file}: {row}/{total_rows} traités ({progress:.1f}%)"
                    
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
                        results = self.serialize_hit_tuple(hit_tuple)
                        list_hits.append(results)
                    match_factor_min = 650 #TODO parametre ??
                    top_hits = self.filter_best_hits(list_hits, match_factor_min)

                    def join_field(field):
                        return '/'.join(str(m.get(field, '')) for m in top_hits)

                    if top_hits:
                        compounds_identified += 1
                        identification_data_dict = {
                            'compound_name': join_field('name'),
                            'casno': join_field('cas_number'),
                            'compound_formula': join_field('formula'),
                            'match_factor': join_field('match_factor'),
                        }
                        for key in identification_data_dict:
                            df.at[row, key] = identification_data_dict[key]

                output_filepath = os.path.join(output_path, f"identified_{file}")
                df.to_csv(output_filepath, sep=";", index=False, encoding="utf-8-sig") #compatibilite avec excel
                yield f"? {file}: terminé"
        
        yield f"?? Temps total: {time.time()-start:.2f} secondes"
    
    # Dans ta route Flask
        for message in self.matching_nist(input_path, output_path, files_list):
            yield f"data: {json.dumps({'type': 'message', 'content': message, 'message_type': 'info'})}\n\n"

    def matching_nist2(self, input_path, output_path, files_list):
        """
        Perform a local NIST search operation.

        :param input_path: Path to the input data.
        :param output_path: Path to save the output results.
        :param files: List of files to process.
        :return: Search results.
        """
        messages = []
        
        #messages.append("🔬 Starting NIST local search...")
        start = time.time()
        
      #  files_list = self.check_files(input_path, files, messages)

        with pyms_nist_search.Engine(
                self.mainlib_path,
                pyms_nist_search.NISTMS_MAIN_LIB,
                self.temp_dir) as engine:
            
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
                        results = self.serialize_hit_tuple(hit_tuple)
                        list_hits.append(results)
                    match_factor_min = 650 #TODO parametre ??
                    top_hits = self.filter_best_hits(list_hits, match_factor_min)

                    def join_field(field):
                        return '/'.join(str(m.get(field, '')) for m in top_hits)

                    if top_hits:
                        compounds_identified += 1
                        identification_data_dict = {
                            'compound_name': join_field('name'),
                            'casno': join_field('cas_number'),
                            'compound_formula': join_field('formula'),
                            'match_factor': join_field('match_factor'),
                        }
                        for key in identification_data_dict:
                            df.at[row, key] = identification_data_dict[key]

                output_filepath = os.path.join(output_path, f"identified_{file}")
                df.to_csv(output_filepath, sep=";", index=False, encoding="utf-8-sig") #compatibilite avec excel
                messages.append(f"✅ {file}: {compounds_identified}/{compounds_processed} composés identifiés")
                messages.append(f"💾 Fichier CSV identifié sauvegardé: {output_filepath}")

        end = time.time() - start
        messages.append(f"⏱️ NIST local search completed in {end:.2f} secondes")

        return messages

