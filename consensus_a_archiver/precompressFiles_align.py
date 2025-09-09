from tkinter import TRUE
from turtle import write
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ChromatographicPrecompressFiles:
    """This Class is an optional pre-processing step before running consensus
    align to identify peaks that likely need to be combined prior to running
    consensus align and will perform a rough combine of these peaks depending
    on the quant method as an output"""

    def __init__(self, rt1_penalty=1,
                 rt2_penalty=10,
                 similarity_cutoff=95,
                 num_cores=1,
                 common_ions=None,
                 quant_method="T",
                 output_files=False):
        self.rt1_penalty = rt1_penalty
        self.rt2_penalty = rt2_penalty
        self.similarity_cutoff = similarity_cutoff
        self.num_cores = num_cores
        self.common_ions = common_ions
        self.quant_method = quant_method
        self.output_files = output_files

    def importFile(self, file):
        """Import and process chromatographic data file
        Parameters:
        -----------
        file : str
            Path to the chromatographic data file
            
        Returns:
        --------
        list : [dataframe, spectra_list, missing_standards, ion_names, spectra_split]
        """

        #read the file    
        current_raw_file = pd.read_csv(file, sep="\t", header=0, skipinitialspace=True)
        current_raw_file = current_raw_file.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))


        # Convert columns to string
        current_raw_file.iloc[:, 4] = current_raw_file.iloc[:, 4].astype(str)
        current_raw_file.iloc[:, 1] = current_raw_file.iloc[:, 1].astype(str)

        rt_split = current_raw_file.iloc[:, 1].str.replace('"', '', regex=False).str.split(' , ', expand=True)
        current_raw_file["RT1"] = rt_split[0].astype(float)
        current_raw_file["RT2"] = rt_split[1].astype(float)

        # Create unique index for duplicate removal (Name + RT + Area)
        unique_index = (current_raw_file.iloc[:, 0].astype(str) + 
                    current_raw_file.iloc[:, 1].astype(str) + 
                    current_raw_file.iloc[:, 2].astype(str))
        
        # Remove duplicates
        current_raw_file = current_raw_file.loc[~unique_index.duplicated()].reset_index(drop=True)

        # Parse spectra data
        spectra_split = []
        ion_names = None

        #     spectra_split.append(np.array(sorted_intensities))
        for i, row in current_raw_file.iterrows():
            spectrum = row.iloc[4]  # Column 5 in R = index 4 in Python
            if pd.isna(spectrum) or spectrum == '':
                spectra_split.append(np.array([]))
                continue
                
            # Split spectrum by spaces, then by colons
            peak_list = []
            for peak in spectrum.strip().split(" "):
                if ":" in peak:
                    parts = peak.split(":")
                    if len(parts) == 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peak_list.append((mz, intensity))
                        except ValueError:
                            continue
            
            if peak_list:
                # Sort by m/z
                peak_list.sort(key=lambda x: x[0])
                mzs, intensities = zip(*peak_list)
                
                # Set ion_names from first spectrum
                if ion_names is None:
                    ion_names = list(mzs)
                
                spectra_split.append(np.array(intensities))
            else:
                spectra_split.append(np.array([]))

        return [current_raw_file, spectra_split, ion_names]


    def find_matches(self, sample):
        """
        Calculate pairwise similarity scores between all metabolite spectra,
        apply retention time penalties, and return indices of similar metabolites.
        
        Parameters:
            sample: tuple (dataframe, list of spectra, ionNames)
            RT1Penalty: float, penalty factor for RT1 differences
            RT2Penalty: float, penalty factor for RT2 differences
            similarityCutoff: float, threshold for similarity

        Returns:
            list of lists of matching indices (1-based, like R)
        """
        df, spectra_list, _ = sample
        spectra_matrix = np.column_stack(spectra_list).T  # transpose to match R cbind + t
        # normalize each row (like R)
        norms = np.linalg.norm(spectra_matrix, axis=1, keepdims=True)
        spectra_matrix = spectra_matrix / norms

        # similarity matrix
        sim_matrix = np.dot(spectra_matrix, spectra_matrix.T) * 100

        # RT1 and RT2 penalties
        RT1 = df['RT1'].to_numpy()
        RT2 = df['RT2'].to_numpy()
        RT1_penalty = np.abs(RT1[:, None] - RT1[None, :]) * self.rt1_penalty
        RT2_penalty = np.abs(RT2[:, None] - RT2[None, :]) * self.rt2_penalty

        sim_matrix = sim_matrix - RT1_penalty - RT2_penalty

        # zero out diagonal
        np.fill_diagonal(sim_matrix, 0)

        # find matches above threshold
        match_list = []
        for row in sim_matrix:
            matches = np.where(row >= self.similarity_cutoff)[0]
            match_list.append((matches + 1).tolist() if len(matches) > 0 else [])
             # match_list.append(matches.tolist()) # TODO modif ici, NON fonctionnel
        return match_list
    
    def parse_spectrum(self, spectrum_str):
        """
        Parse un spectre du format 'ion:intensité ...'
        Exclut les ions communs.
        """
        pairs = [s.split(":") for s in spectrum_str.split()]
        arr = np.array([[float(mz), float(intensity)] for mz, intensity in pairs])
        # arr = arr[~np.isin(arr[:, 0], common_ions)]
        arr = arr[np.argsort(arr[:, 0])]
        return arr[:, 1]


   # TODO version fonctionnelle ms mate  en pairwise comme en R 
    def precompress_files(self, input_file_list):
        if self.common_ions is None:
            common_ions = []
        combined_list = {}

        # utilisation de multiprocessing.pool est plus proche de mclapply en R et plus simple. pas besoin de ProcessPoolExecutor ici (plus puissant, gestion d erreur,...)
        with Pool(processes=self.num_cores) as pool:
            imported_files = pool.map(self.importFile, input_file_list)

        with Pool(processes=self.num_cores) as pool:
            match_list = pool.map(self.find_matches, imported_files)

        for samp_num, matches in enumerate(match_list):
            num_reps = 0
            if len(matches) > 0:
                num_reps = max(len(m) for m in matches) - 1
                if self.quant_method in ["T"]:
                    mates = [x[0] - 1 if len(x) > 0 else None for x in matches]
                    binding_areas = imported_files[samp_num][0].iloc[
                        [m for m in mates if m is not None], 2
                    ].values

                    # Find mates partners to combine
                    to_bind = imported_files[samp_num][0].iloc[
                        [i for i, m in enumerate(mates) if m is not None], :
                    ].copy()
                    # Filtrer les mates valides #TODO MODIF
                    valid_mates = [m for m in mates if m is not None and not pd.isna(m)]
                    # print("valid_mates", valid_mates)

                    # Sélectionner les lignes correspondantes
                    mates_df = imported_files[samp_num][0].iloc[valid_mates, :].reset_index(drop=True)
                    n_mates = len(valid_mates)
                    if len(to_bind) == 1:
                        to_bind_repeated = pd.concat([to_bind]*n_mates, ignore_index=True)
                    else:
                        # Si to_bind a plusieurs lignes, il doit correspondre exactement à n_mates
                        to_bind_repeated = to_bind.reset_index(drop=True).iloc[:n_mates, :]

                    # Ajouter la colonne source
                    source_series = pd.Series([input_file_list[samp_num]] * len(valid_mates), name="source")

                    # Concaténation finale
                    combined_list[input_file_list[samp_num]] = pd.concat([to_bind_repeated, mates_df, source_series], axis=1)
                    if samp_num == 0:
                        combined_list_df = pd.DataFrame(combined_list[input_file_list[samp_num]])
                        combined_list_df.to_csv("py_combined_list.csv", sep="\t", index=False)


                    current_df = imported_files[samp_num][0].copy()
                    for i, m in enumerate(mates):
                        if m is not None:
                            area_i = current_df.iloc[i, 2]
                            area_m = current_df.iloc[m, 2]
                            print(f"[COMBINE] Peak {i+1} (area={area_i}) + Peak {m+1} (area={area_m}) -> {area_i + area_m}")
                    # Sum peak areas
                    to_bind.loc[:, to_bind.columns[2]] += binding_areas
                    # Ensure only one peak combination gets included in output
                    to_bind["Bound"] = [
                        f"{min(m, i)}"
                        for i, m in enumerate(mates) if m is not None
                    ]

                    to_bind = to_bind.drop_duplicates(subset=["Bound"])

                    # Update sample metabolite file to include on combined peak
                    imported_files[samp_num][0] = pd.concat([
                        imported_files[samp_num][0].iloc[
                            [i for i, m in enumerate(mates) if m is None], :
                            ], to_bind.drop(columns=["Bound"])
                            ])
                    
            #If any metabolites had greater than two peaks to combine, loop through and make those combinations iteratively
            if num_reps > 0:
                original_df = imported_files[samp_num][0].copy()
                for rep in range(num_reps):
    
                    #Repeat similarity scores with combined peaks
                    df = imported_files[samp_num][0].copy()
                    spectra_split = [self.parse_spectrum(row.iloc[4]) for _, row in df.iterrows()]
                    spectra_matrix = np.vstack([
                        vec / np.sqrt(np.sum(vec ** 2)) for vec in spectra_split
                    ])
                    similarity_matrix = (spectra_matrix @ spectra_matrix.T) * 100

                    #Subtract retention time difference penalties from similarity scores
                    rt1 = df["RT1"].values
                    rt2 = df["RT2"].values
                    RT1Index = np.abs(rt1[:, None] - rt1[None, :]) * self.rt1_penalty
                    RT2Index = np.abs(rt2[:, None] - rt2[None, :]) * self.rt2_penalty
                    similarity_matrix = similarity_matrix - RT1Index - RT2Index
                    np.fill_diagonal(similarity_matrix, 0)
            
                    #Repeat peak combination if more combinations are necessary
                    new_matches = [np.where(row >= self.similarity_cutoff)[0] for row in similarity_matrix]
                    if any(len(arr) > 0 for arr in new_matches):
                        if self.quant_method == "T":
                            mates = [m[0]  if len(m) > 0 else None for m in new_matches] #TODO modif ici -1
                            binding_areas = df.iloc[
                                [m for m in mates if m is not None], 2
                            ].values
                            to_bind = df.iloc[
                                [i for i, m in enumerate(mates) if m is not None], :
                            ].copy()

                            valid_mates = [m for m in mates if m is not None and not pd.isna(m)]
                            n_mates = len(valid_mates)
                            print("valid_mates 2", valid_mates)

                            # DataFrame des mates
                            mates_df = df.iloc[valid_mates, :].reset_index(drop=True)
                            # print("mates_df", mates_df)

                            # S'assurer que to_bind a le même nombre de lignes que mates_df
                            if len(to_bind) == 1:
                                # répéter to_bind pour chaque mate
                                to_bind_repeated = pd.concat([to_bind]*n_mates, ignore_index=True)
                            else:
                                # Si to_bind a plusieurs lignes, on coupe ou on garde les n_mates premières lignes
                                to_bind_repeated = to_bind.reset_index(drop=True).iloc[:n_mates, :]

                            # Ajouter colonne source
                            source_series = pd.Series([input_file_list[samp_num]] * n_mates, name="source")

                            new_combined = pd.concat([to_bind_repeated, mates_df], axis=1)
                            new_combined["source"] = input_file_list[samp_num]

                            # --- Ajouter au combined_list existant ---
                            if input_file_list[samp_num] not in combined_list:
                                combined_list[input_file_list[samp_num]] = pd.DataFrame()  # initialisation
                            combined_list[input_file_list[samp_num]] = pd.concat(
                                [combined_list[input_file_list[samp_num]], new_combined],
                                ignore_index=True
                            )


                            for i, m in enumerate(mates):
                                if m is not None:
                                    area_i = df.iloc[i, 2]
                                    area_m = df.iloc[m, 2]
                                    print(f"[REP {rep}] Peak {i+1} (area={area_i}) + Peak {m+1} (area={area_m}) -> {area_i + area_m}")
                            # --- Mettre à jour to_bind avec les aires combinées ---
                            binding_areas = df.iloc[valid_mates, 2].values
                            to_bind.loc[:, to_bind.columns[2]] += binding_areas
                            to_bind["Bound"] = [f"{min(m, i)}" for i, m in enumerate(mates) if m is not None]
                            to_bind = to_bind.drop_duplicates(subset=["Bound"])

                            # --- Remplacer le DataFrame original avec les pics non liés + pics combinés ---
                            imported_files[samp_num][0] = pd.concat([
                                df.iloc[[i for i, m in enumerate(mates) if m is None], :],
                                to_bind.drop(columns=["Bound"])
                            ])
                        else:
                            # Pas de mates : on garde juste le DataFrame original
                            if input_file_list[samp_num] not in combined_list:
                                combined_list[input_file_list[samp_num]] = df.copy()
                            combined_list[input_file_list[samp_num]]["source"] = input_file_list[samp_num]

                    else:
                        # Pas de mates : garder le DataFrame original
                        if input_file_list[samp_num] not in combined_list:
                            combined_list[input_file_list[samp_num]] = original_df.copy()
                        combined_list[input_file_list[samp_num]]["source"] = input_file_list[samp_num]


        #Make data frame with all combined peak pair info
        if len(combined_list) > 0:
            combined_frame = pd.concat(combined_list.values(), ignore_index=True)
        else:
            combined_frame = pd.DataFrame()

        #If outputFiles==TRUE, write processed files out to the input file directory
        if self.output_files:
            for samp_num, imp in enumerate(imported_files):
                out_name = (
                    input_file_list[samp_num][:-4] + "_Py_Processed_last.csv"
                )
                imp[0].iloc[:, :5].to_csv(out_name, sep="\t", index=False)

        return combined_frame

#all mates ;  regroupement  transitif
    # def precompress_files(self, input_file_list):
    #     if self.common_ions is None:
    #         common_ions = []
        
    #     combined_list = {}

    #     # --- Import des fichiers en parallèle ---
    #     with Pool(processes=self.num_cores) as pool:
    #         imported_files = pool.map(self.importFile, input_file_list)

    #     # --- Calcul des matchs spectres ---
    #     with Pool(processes=self.num_cores) as pool:
    #         match_list = pool.map(self.find_matches, imported_files)

    #     for samp_num, matches in enumerate(match_list):
    #         if len(matches) == 0:
    #             continue
            
    #         # --- Crée la liste de tous les mates pour chaque peak ---
    #         # mates_list = [ [i - 1 for i in x] if len(x) > 0 else [] for x in matches ]

    #         print("matches:", matches)
    #         # print("mates_list", mates_list)
    #         # print("Matches for peak 287:", matches[287])

    #         df = imported_files[samp_num][0].copy()
    #         used_peaks = set()
    #         combined_rows = []

    #         # for peak_idx, mates in enumerate(mates_list):
    #         # for peak_idx, mates in enumerate(matches):
    #         #     if peak_idx in used_peaks:
    #         #         continue
    #         #     # Inclure le peak lui-même
    #         #     all_indices = set([peak_idx] + mates)
    #         #     # Ajouter récursivement tous les mates de ces mates
    #         #     queue = list(all_indices)
    #         #     while queue:
    #         #         idx = queue.pop()
    #         #         if idx not in all_indices:
    #         #             all_indices.add(idx)
    #         #         # for m in mates_list[idx]:
    #         #         for m in matches[idx]:
    #         #             if m not in all_indices:
    #         #                 all_indices.add(m)
    #         #                 queue.append(m)
    #         for peak_idx, mates in enumerate(matches):
    #             if peak_idx in used_peaks:
    #                 continue

    #             # Inclure le peak lui-même
    #             all_indices = set([peak_idx] + mates)

    #             # Ajouter récursivement tous les mates de ces mates
    #             queue = list(all_indices)
    #             while queue:
    #                 idx = queue.pop()
    #                 if idx < 0 or idx >= len(matches):
    #                     continue  # protection contre indices hors limites
    #                 for m in matches[idx]:
    #                     if 0 <= m < len(df) and m not in all_indices:
    #                         all_indices.add(m)
    #                         queue.append(m)
            

    #             used_peaks.update(all_indices)
    #             # all_indices = sorted(all_indices)
    #             all_indices = sorted([i for i in all_indices if i < len(df)])  # sécurité


    #             # Calcul de la somme des aires
    #             if len(all_indices) == 1:
    #                 combined_row = df.iloc[all_indices[0]].copy()
    #                 combined_area = combined_row.iloc[2]  # aire inchangée
    #             else:
    #                 combined_area = df.iloc[all_indices, 2].sum()
    #                 combined_row = df.iloc[all_indices[0]].copy()
    #                 combined_row.iloc[2] = combined_area
    #             # Somme des aires
    #             # combined_area = df.iloc[all_indices, 2].sum()
    #             # combined_row = df.iloc[all_indices[0]].copy()
    #             # combined_row.iloc[2] = combined_area  # mettre la somme dans la colonne aire

    #             # Concaténer les mates dans une colonne Bound
    #             combined_row["Bound"] = "/".join(str(i+1) for i in all_indices)
    #             # print("Combined peaks:", combined_row["Bound"])
    #             # print(f"Combined peaks: {combined_row['Bound']} | Combined area: {combined_row.iloc[2]}")

    #             combined_rows.append(combined_row)

    #         combined_df = pd.DataFrame(combined_rows)
    #         combined_df["source"] = input_file_list[samp_num]
    #         combined_list[input_file_list[samp_num]] = combined_df

    #         # Mettre à jour le DataFrame original pour inclure les pics combinés
    #         imported_files[samp_num][0] = combined_df.drop(columns=["Bound", "source"])

    #     # --- Concaténer tous les fichiers ---
    #     if len(combined_list) > 0:
    #         combined_frame = pd.concat(combined_list.values(), ignore_index=True)
    #     else:
    #         combined_frame = pd.DataFrame()

    #     # --- Écrire les fichiers traités si demandé ---
    #     if self.output_files:
    #         for samp_num, imp in enumerate(imported_files):
    #             out_name = input_file_list[samp_num][:-4] + "_Py_Processed_tot.csv"
    #             imp[0].iloc[:, :5].to_csv(out_name, sep="\t", index=False)

    #     return combined_frame
    
    # def precompress_files(self, input_file_list):
    #     combined_list = {}

    #     # --- Import des fichiers en parallèle ---
    #     with Pool(processes=self.num_cores) as pool:
    #         imported_files = pool.map(self.importFile, input_file_list)

    #     # --- Calcul des matchs spectres ---
    #     with Pool(processes=self.num_cores) as pool:
    #         match_list = pool.map(self.find_matches, imported_files)

    #     for samp_num, matches in enumerate(match_list):
    #         if len(matches) == 0:
    #             continue

    #         df = imported_files[samp_num][0].copy()
    #         used_peaks = set()
    #         combined_rows = []

    #         for peak_idx, mates in enumerate(matches):
    #             if peak_idx in used_peaks or len(mates) == 0:
    #                 # Pic déjà traité ou pic isolé → ignorer
    #                 continue

    #             # Inclure le peak lui-même
    #             all_indices = set([peak_idx] + mates)

    #             # Ajouter récursivement tous les mates de ces mates
    #             queue = list(all_indices)
    #             while queue:
    #                 idx = queue.pop()
    #                 if idx < 0 or idx >= len(matches):
    #                     continue  # protection contre indices hors limites
    #                 for m in matches[idx]:
    #                     if 0 <= m < len(df) and m not in all_indices:
    #                         all_indices.add(m)
    #                         queue.append(m)

    #             used_peaks.update(all_indices)
    #             all_indices = sorted([i for i in all_indices if i < len(df)])  # sécurité

    #             # Somme des aires
    #             combined_area = df.iloc[all_indices, 2].sum()
    #             combined_row = df.iloc[all_indices[0]].copy()
    #             combined_row.iloc[2] = combined_area

    #             # Concaténer les mates dans Bound
    #             combined_row["Bound"] = "/".join(str(i+1) for i in all_indices)  # +1 si besoin pour compat R
    #             print(f"Combined peaks: {combined_row['Bound']} | Combined area: {combined_area}")

    #             combined_rows.append(combined_row)

    #         if combined_rows:  # n’ajouter que si au moins un pic combiné
    #             combined_df = pd.DataFrame(combined_rows)
    #             combined_df["source"] = input_file_list[samp_num]
    #             combined_list[input_file_list[samp_num]] = combined_df

    #         # Mettre à jour le DataFrame original pour inclure uniquement les pics non combinés
    #         non_combined_df = df.drop(index=list(used_peaks))
    #         imported_files[samp_num][0] = non_combined_df

    #     # --- Concaténer tous les fichiers combinés ---
    #     if combined_list:
    #         combined_frame = pd.concat(combined_list.values(), ignore_index=True)
    #     else:
    #         combined_frame = pd.DataFrame()

    #     # --- Écrire les fichiers traités si demandé ---
    #     if self.output_files:
    #         for samp_num, imp in enumerate(imported_files):
    #             out_name = input_file_list[samp_num][:-4] + "_Py_Processed_tot.csv"
    #             imp[0].iloc[:, :5].to_csv(out_name, sep="\t", index=False)

    #     return combined_frame

    # def precompress_files(self, input_file_list):
    #     combined_list = {}

    #     # --- Import des fichiers en parallèle ---
    #     with Pool(processes=self.num_cores) as pool:
    #         imported_files = pool.map(self.importFile, input_file_list)

    #     for samp_num, imported in enumerate(imported_files):
    #         df = imported[0].copy()
    #         matches = self.find_matches(imported)  # matches pour ce fichier
    #         used_peaks = set()
    #         combined_rows = []

    #         # --- Boucle sur chaque pic pour combiner les mates ---
    #         for peak_idx, mates in enumerate(matches):
    #             if peak_idx in used_peaks or len(mates) == 0:
    #                 continue  # pic déjà combiné ou pic isolé

    #             # --- Normalisation si matches 1-based (R) ---
    #             mates_corrected = [m - 1 for m in mates]  # supprimer si matches déjà 0-based

    #             # Inclure le pic lui-même
    #             all_indices = set([peak_idx] + mates_corrected)

    #             # Propagation récursive des mates
    #             queue = list(all_indices)
    #             while queue:
    #                 idx = queue.pop()
    #                 if idx < 0 or idx >= len(matches):
    #                     continue
    #                 for m in matches[idx]:
    #                     m_corr = m - 1  # ajustement indices
    #                     if 0 <= m_corr < len(df) and m_corr not in all_indices:
    #                         all_indices.add(m_corr)
    #                         queue.append(m_corr)

    #             used_peaks.update(all_indices)
    #             all_indices = sorted([i for i in all_indices if i < len(df)])

    #             # --- Création de la ligne combinée ---
    #             combined_area = df.iloc[all_indices, 2].sum()
    #             combined_row = df.iloc[all_indices[0]].copy()
    #             combined_row.iloc[2] = combined_area
    #             combined_row["Bound"] = "/".join(str(i+1) for i in all_indices)
    #             combined_rows.append(combined_row)
    #             print(f"Combined peaks: {combined_row['Bound']} | Combined area: {combined_area}")

    #         # --- Créer DataFrame des pics combinés si au moins un pic ---
    #         if combined_rows:
    #             combined_df = pd.DataFrame(combined_rows)
    #             combined_df["source"] = input_file_list[samp_num]
    #             combined_list[input_file_list[samp_num]] = combined_df

    #         # --- Pics isolés : ceux non combinés, inchangés ---
    #         remaining_df = df.drop(index=list(used_peaks))
    #         imported_files[samp_num][0] = remaining_df

    #     # --- Concaténer tous les pics combinés pour py_combined_frame.csv ---
    #     if combined_list:
    #         combined_frame = pd.concat(combined_list.values(), ignore_index=True)
    #     else:
    #         combined_frame = pd.DataFrame()

    #     # --- Écriture des fichiers ---
    #     if self.output_files:
    #         # Pics isolés
    #         for samp_num, imp in enumerate(imported_files):
    #             out_name = input_file_list[samp_num][:-4] + "_Py_Processed.csv"
    #             imp[0].iloc[:, :5].to_csv(out_name, sep="\t", index=False)

    #         # Pics combinés
    #         if not combined_frame.empty:
    #             out_combined = "py_combined_frame.csv"
    #             combined_frame.to_csv(out_combined, sep="\t", index=False)

    #     return combined_frame



#version claude a tester
    # def precompress_files(self, input_file_list):
    #     combined_list = {}
        
    #     with Pool(processes=self.num_cores) as pool:
    #         imported_files = pool.map(self.importFile, input_file_list)
        
    #     for samp_num, imported in enumerate(imported_files):
    #         df = imported[0].copy()
    #         matches = self.find_matches(imported)
            
    #         # Créer le DataFrame final qui contiendra tous les pics (combinés + isolés)
    #         final_df = df.copy()
    #         used_peaks = set()
    #         combined_rows = []
    #         rows_to_remove = []  # indices à supprimer du DataFrame final
            
    #         # Traitement des pics avec mates
    #         for peak_idx, mates in enumerate(matches):
    #             if peak_idx in used_peaks or len(mates) == 0:
    #                 continue
                
    #             # Collecte transitive des indices
    #             all_indices = set([peak_idx])
    #             queue = [peak_idx]
                
    #             while queue:
    #                 current_idx = queue.pop()
    #                 if current_idx < 0 or current_idx >= len(matches):
    #                     continue
                        
    #                 for mate in matches[current_idx]:
    #                     mate_idx = mate - 1  # correction si 1-based
    #                     if (0 <= mate_idx < len(df) and mate_idx not in all_indices):
    #                         all_indices.add(mate_idx)
    #                         queue.append(mate_idx)
                
    #             # Si on a plusieurs pics à combiner
    #             if len(all_indices) > 1:
    #                 used_peaks.update(all_indices)
    #                 all_indices = sorted(list(all_indices))
                    
    #                 # Création du pic combiné
    #                 combined_area = df.iloc[all_indices, 2].sum()
    #                 combined_row = df.iloc[all_indices[0]].copy()
    #                 combined_row.iloc[2] = combined_area
    #                 combined_row["Bound"] = "/".join(str(i+1) for i in all_indices)
    #                 combined_rows.append(combined_row)
                    
    #                 # Remplacer le premier pic par le pic combiné dans final_df
    #                 final_df.iloc[all_indices[0]] = combined_row
                    
    #                 # Marquer les autres pics pour suppression
    #                 rows_to_remove.extend(all_indices[1:])
                    
    #                 print(f"Combined peaks: {combined_row['Bound']} | Combined area: {combined_area}")
            
    #         # Supprimer les pics qui ont été fusionnés (sauf le premier de chaque groupe)
    #         if rows_to_remove:
    #             final_df = final_df.drop(index=rows_to_remove).reset_index(drop=True)
            
    #         # Stocker les informations de combinaison
    #         if combined_rows:
    #             combined_df = pd.DataFrame(combined_rows)
    #             combined_df["source"] = input_file_list[samp_num]
    #             combined_list[input_file_list[samp_num]] = combined_df
            
    #         # Écriture du fichier final (pics combinés + pics isolés)
    #         if self.output_files:
    #             out_name = input_file_list[samp_num][:-4] + "_Py_Processed.csv"
    #             final_df.iloc[:, :5].to_csv(out_name, sep="\t", index=False)
        
    #     # Concaténation de tous les pics combinés
    #     if combined_list:
    #         combined_frame = pd.concat(combined_list.values(), ignore_index=True)
    #         if self.output_files:
    #             combined_frame.to_csv("py_combined_frame.csv", sep="\t", index=False)
    #     else:
    #         combined_frame = pd.DataFrame()
        
    #     return combined_frame


if __name__ == "__main__":
    listFiles = [
        #"/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE.txt",
        #"/home/camille/Documents/app/data/cdf et h5/new/peak_detection/2025-04-10-854514_Q.txt",
        "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751310_0048GL_M1_postPTR_split.txt",
        #"/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751315_0033CN_J7_postPTR_split.txt"
        ]

#test find_matches
    # precompressedFiles = ChromatographicPrecompressFiles(rt1_penalty=1, rt2_penalty=10, similarity_cutoff=90)
    # ImportedFiles = [precompressedFiles.importFile(file) for file in listFiles] 
    # for samp_num in range(len(ImportedFiles)):
    #     match_list = precompressedFiles.find_matches(ImportedFiles[samp_num])
    #         # Transformation en chaîne de caractères
    #     match_list_str = []
    #     for x in match_list:
    #         if len(x) == 0:
    #             match_list_str.append("no_match")         # cas vide
    #         elif all(isinstance(i, bool) for i in x):
    #             match_list_str.append(",".join(map(str, x)))  # cas True/False
    #         else:
    #             match_list_str.append(",".join(map(str, x)))  # cas normal

    #     # Création du DataFrame et écriture
    #     df = pd.DataFrame({"MatchList": match_list_str})
    #     output_file = f"/home/camille/Documents/app/data/output/py_MatchList_{samp_num+1}.txt"
    #     df.to_csv(output_file, sep="\t", index=False, header=False, quoting=3)  # quoting=3 pour pas de guillemets


#test precompressFiles
    precompressedFiles = ChromatographicPrecompressFiles(rt1_penalty=1, rt2_penalty=10, similarity_cutoff=90, num_cores=1, common_ions=None, quant_method="T", output_files=True)
    combined_frame = precompressedFiles.precompress_files(listFiles)
    # print(combined_frame)

    # precompressedFiles.plot_combined_peaks(combined_frame, "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE.txt").
    combined_frame_df = pd.DataFrame(combined_frame)
    combined_frame_df.to_csv("py_combined_frame_last.csv", sep="\t", index=False)