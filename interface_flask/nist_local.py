import pyms_nist_search
import os
import pyms.Spectrum
from dotenv import load_dotenv
import time
import pandas as pd

load_dotenv()


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


def matching_nist_streaming(input_path, output_path, files, progress_callback=None):
    """Version streaming de matching_nist"""
    messages = []
    start = time.time()
    
    def send_message(msg, msg_type='info'):
        if progress_callback:
            list(progress_callback(msg, msg_type))  # Envoyer immédiatement
        messages.append(msg)
    
    send_message("🔬 Starting NIST local search...")
    
    mainlib_path = os.getenv("MAINLIB_PATH", "C:/NIST20/MSSEARCH/mainlib")
    temp_dir = os.getenv("TEMP_DIR", "C:/NIST20/MSSEARCH/tmp")

    files_list = [f.strip() for f in files.split(',') if f.strip()]
    peak_info_files = [f for f in files_list if "peak_info" in f.lower()]
    
    if not peak_info_files:
        send_message("⚠️ Aucun fichier peak_info trouvé", 'warning')
        return messages
    
    send_message(f"📋 Fichiers peak_info à traiter: {peak_info_files}")
    
    with pyms_nist_search.Engine(mainlib_path, pyms_nist_search.NISTMS_MAIN_LIB, temp_dir) as engine:
        
        for file_idx, file in enumerate(peak_info_files):
            filepath = os.path.join(input_path, file)
            send_message(f"🔬 Processing file {file_idx+1}/{len(peak_info_files)}: {file}")
            
            try:
                df = pd.read_csv(filepath, sep=";")
                df['compound_name'] = ""
                df['casno'] = ""
                df['compound_formula'] = ""
                df['match_factor'] = ""
                
                compounds_processed = 0
                compounds_identified = 0
                total_rows = df.shape[0]

                for row in range(total_rows):
                    try:
                        # ✅ Message de progression tous les 10 composés
                        if row % 10 == 0:
                            progress_pct = (row / total_rows) * 100
                            send_message(f"📊 Fichier {file}: {row}/{total_rows} composés traités ({progress_pct:.1f}%)")
                        
                        s = df.at[row, 'Spectra']
                        pairs = s.strip().split()

                        masses = []
                        intensities = []

                        for pair in pairs:
                            m, i = pair.split(":")
                            masses.append(int(float(m)))
                            intensities.append(float(i))

                        compounds_processed += 1
                        
                        if not masses:
                            continue

                        list_hits = []
                        mass_spectrum = pyms.Spectrum.MassSpectrum(masses, intensities)

                        hits = engine.full_search_with_ref_data(mass_spectrum, n_hits=20)
                        for i, hit_tuple in enumerate(hits):
                            results = serialize_hit_tuple(hit_tuple)
                            list_hits.append(results)
                            
                        match_factor_min = 650
                        top_hits = filter_best_hits(list_hits, match_factor_min)

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
                                
                    except Exception as e:
                        send_message(f"⚠️ Erreur ligne {row}: {str(e)}", 'warning')
                        continue

                # ✅ Sauvegarde
                output_filepath = os.path.join(output_path, f"identified_{file}")
                df.to_csv(output_filepath, sep=";", index=False, encoding="utf-8-sig")
                
                send_message(f"🎯 {file}: {compounds_identified}/{compounds_processed} composés identifiés", 'success')
                send_message(f"💾 Sauvegardé: {output_filepath}", 'success')
                
            except Exception as e:
                send_message(f"❌ Erreur fichier {file}: {str(e)}", 'error')
                continue

    end = time.time() - start
    send_message(f"⏱️ Temps total: {end:.2f} secondes", 'success')

    return messages


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
    peak_info_files = [f for f in files_list if "peak_info" in f.lower()]

    if not peak_info_files:
        messages.append("⚠️ Aucun fichier 'peak_info' trouvé dans la liste")
        return messages
    messages.append(f"📋 Fichiers à traiter: {files_list}")

    with pyms_nist_search.Engine(
            mainlib_path,
            pyms_nist_search.NISTMS_MAIN_LIB,
            temp_dir) as engine:
        
        for file in peak_info_files:
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
                    results = serialize_hit_tuple(hit_tuple)
                    list_hits.append(results)
                match_factor_min = 650 #TODO parametre ??
                top_hits = filter_best_hits(list_hits, match_factor_min)

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

