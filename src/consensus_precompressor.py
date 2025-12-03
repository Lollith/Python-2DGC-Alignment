import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import os
import logging

class PeakPrecompressor:
    """This Class is an optional pre-processing step before running consensus
    align to identify peaks that likely need to be combined prior to running
    consensus align and will perform a rough combine of these peaks depending
    on the quant method as an output"""

    def __init__(self,
                 rt1_penalty=1,
                 rt2_penalty=10,
                 similarity_cutoff=95,
                 num_cores=1,
                 common_ions=None,
                 quant_method="T",
                 #area_selection="area_mod_max",
                ):
        self.rt1_penalty = rt1_penalty
        self.rt2_penalty = rt2_penalty
        self.similarity_cutoff = similarity_cutoff
        self.num_cores = num_cores
        self.common_ions = common_ions
        self.quant_method = quant_method
        #self.area_selection = area_selection
        self.docker_volume_path = os.environ.get("DOCKER_VOLUME_PATH", "/app/data/")
        self.logger = logging.getLogger('gcgc_cli')

    def importFile(self, file):
        """Import and process chromatographic data file
        Parameters:
        -----------
        file : str
            Path to the chromatographic data file
            
        Returns:
        --------
        list : [dataframe, spectra_list, missing_standards, fion_names, spectra_split]
        """

        #read the file    
        current_raw_file = pd.read_csv(file, sep="\t", header=0, skipinitialspace=True)
        current_raw_file = current_raw_file.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        
        # verifie si .txt contient aire deconvolution + aire mod_max ou juste 1 aire
        #current_raw_file, spectra_col_index = self.check_file_parameters(current_raw_file)

        # Convert columns to string
        current_raw_file.iloc[:, 5] = current_raw_file.iloc[:, 5].astype(str)
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
            spectrum = row.iloc[5]  # Column 5 in R = index 4 in Python
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

    def precompress_files(self, input_file_list, output_dir):
        # print(f"Precompressing {len(input_file_list)} files...", flush=True)
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
                    binding_areas_deconv = imported_files[samp_num][0].iloc[
                        [m for m in mates if m is not None], 2
                    ].values

                    binding_areas_mod_max = imported_files[samp_num][0].iloc[
                        [m for m in mates if m is not None], 3
                    ].values

                    # Find mates partners to combine
                    to_bind = imported_files[samp_num][0].iloc[
                        [i for i, m in enumerate(mates) if m is not None], :
                    ].copy()
                    valid_mates = [m for m in mates if m is not None and not pd.isna(m)]

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
                    #if samp_num == 0:
                    #    combined_list_df = pd.DataFrame(combined_list[input_file_list[samp_num]])

                    # Debug: afficher les combinaisons effectuées
                   # current_df = imported_files[samp_num][0].copy()
                    #for i, m in enumerate(mates):
                     #   if m is not None:
                      #      area_i = current_df.iloc[i, 2]
                        #    area_m = current_df.iloc[m, 2]
                            # print(f"[COMBINE] Peak {i+1} (area={area_i}) + Peak {m+1} (area={area_m}) -> {area_i + area_m}")
                    # Sum peak areas
                    to_bind.loc[:, to_bind.columns[2]] += binding_areas_deconv
                    to_bind.loc[:, to_bind.columns[3]] += binding_areas_mod_max
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
                    spectra_split = [self.parse_spectrum(row.iloc[5]) for _, row in df.iterrows()]
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
                            mates = [m[0]  if len(m) > 0 else None for m in new_matches]
                            binding_areas_deconvo = df.iloc[
                                [m for m in mates if m is not None], 2
                            ].values
                            binding_areas_mod_max = df.iloc[
                                [m for m in mates if m is not None], 3
                            ].values

                            to_bind = df.iloc[
                                [i for i, m in enumerate(mates) if m is not None], :
                            ].copy()

                            valid_mates = [m for m in mates if m is not None and not pd.isna(m)]
                            n_mates = len(valid_mates)

                            # DataFrame des mates
                            mates_df = df.iloc[valid_mates, :].reset_index(drop=True)

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
                            # --- Mettre à jour to_bind avec les aires combinées ---
                            #binding_areas = df.iloc[valid_mates, 3].values
                            #binding_areas_deconvo = df.iloc[valid_mates, 2].values
                            
                            to_bind.loc[:, to_bind.columns[2]] += binding_areas_deconvo
                            to_bind.loc[:, to_bind.columns[3]] += binding_areas_mod_max
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

        for samp_num, imp in enumerate(imported_files):
            # out_name = (
            #     output_dir + input_file_list[samp_num].split("/")[-1][:-4] + "#P.txt"
            # )
            base_filename = os.path.basename(input_file_list[samp_num])
            output_filename = base_filename.replace('.txt', '#P.txt')
            out_name = os.path.join(output_dir, output_filename)
            imp[0].iloc[:, :6].to_csv(out_name, sep="\t", index=False)
            self.logger.info(f"✅ Fichier {input_file_list[samp_num]} traité, résultat: {out_name}")

        return combined_frame



if __name__ == "__main__":
    listFiles = ["/home/camille/Documents/app/data/output/751303_v3_E3AM_5jui.txt", 
                 "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui.txt"]

#test de find_matches
    # precompressedFiles = PeakPrecompressor(rt1_penalty=1, rt2_penalty=10, similarity_cutoff=90)
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
    precompressedFiles = PeakPrecompressor(rt1_penalty=1, rt2_penalty=10, similarity_cutoff=95, num_cores=1, common_ions=None, quant_method="T", 
                                           #output_files=True
                                           )
    Result = precompressedFiles.precompressFiles(listFiles)
