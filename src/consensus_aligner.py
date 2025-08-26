import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import os
import platform
from datetime import datetime
# import nist_search
import matching

class ChromatographicAligner:
    """
    Class for aligning chromatographic data using consensus alignment.

    Methods:
    --------
    importFile(file):
        Import and process chromatographic data file.
    
    generate_sim_frames(sample, seed_sample, RT2Penalty=5, RT1Penalty=1):
        Generate similarity frames between sample and seed.
    
    consensus_align_bis(input_file_list, ...):
        Main function to perform consensus alignment.
    """

    def __init__(
        self,
        rt1_penalty=1,
        rt2_penalty=10,
        similarity_cutoff=90,
        disimilarity_cutoff=90,
        missing_value_limit=0.75,
        auto_tune_match_stringency=False,
        missing_peak_finder_similarity_lax=0.85,
        quant_method="T",
        num_cores=1,
        nist_api=None
        ):
        """
        Initialize the chromatographic aligner with parameters.
        
        Parameters:
        -----------
        rt1_penalty : int
            Penalty used for first retention time errors.  Defaults to 1.
        rt2_penalty : int
            Penalty used for second retention time errors.  Defaults to 10.
        similarity_cutoff : float
            Adjusts peak similarity threshold required for alignment. 
            Adjust in concordance with RT1 and RT2 penalties. Will 
            be ignored if autoTuneMatchStrigency is TRUE. Defaults to 90.
        disimilarity_cutoff : float
            Defaults to similarityCutoff-90. Sets the threshold for including 
            a new peak in the alignment table to ensure new metabolites aren't
            just below alignment thresholds
        missingValueLimit: float
            Maximum fraction (Numeric between 0 and 1) of missing values 
            acceptable for retaining a metabolite in the final alignment table. 
            Defaults to 0.75.
        num_cores : Number of cores used to parallelize alignment.
            Defaults to 1.
        """
        self.rt1_penalty = rt1_penalty
        self.rt2_penalty = rt2_penalty
        self.similarity_cutoff = similarity_cutoff
        self.disimilarity_cutoff = disimilarity_cutoff
        self.missing_value_limit = missing_value_limit
        self.auto_tune_match_stringency = auto_tune_match_stringency
        self.missing_peak_finder_similarity_lax = missing_peak_finder_similarity_lax
        self.quant_method = quant_method
        self.num_cores = num_cores
        self.nist_api = nist_api

        # results storage
        self.imported_files = None
        self.alignment_results = None
        self.filtered_results = None


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
        missing_standards = []

        #read the file    
        current_raw_file = pd.read_csv(file, sep="\t", header=0,skipinitialspace=True)
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

        return [current_raw_file, spectra_split, missing_standards, ion_names, spectra_split]

    def import_files(self, file_list):
        """Import multiple chromatographic data files
        Parameters:
        -----------
        file_list : list
            List of file paths to import
            
        Returns:
        --------
        list : List of imported file data
        """

        if self.num_cores == 1:
            self.imported_files = [self.importFile(file) for file in file_list]
        else:
            # Ensure compatibility with both Windows and Linux
            if platform.system() == 'Windows':
                import multiprocessing
                multiprocessing.set_start_method('spawn', force=True)
            
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                self.imported_files = list(executor.map(self.importFile, file_list))
        
        return self.imported_files

    def generate_sim_frames(self, sample, seed_sample):
        """Generate similarity matrix between sample and seed.

            Parameters:
        -----------
        sample : list
            Sample chromatographic data
        seed : list
            Seed chromatographic data
            
        Returns:
        --------
        np.ndarray : Similarity matrix with RT penalties applied
        """
        
        # Get spectra data
        seed_spectra_list = seed_sample[1]
        sample_spectra_list = sample[1]
        
        # Convert to matrix and normalize
        max_len = max(len(s) for s in seed_spectra_list + sample_spectra_list if len(s) > 0)
        if max_len == 0:
            max_len = 1
        
        # Pad spectra to same length
        seed_spectra = []
        for spectrum in seed_spectra_list:
            if len(spectrum) == 0:
                padded = np.zeros(max_len)
            elif len(spectrum) < max_len:
                padded = np.pad(spectrum, (0, max_len - len(spectrum)), 'constant')
            else:
                padded = spectrum[:max_len]
            seed_spectra.append(padded)
        
        sample_spectra = []
        for spectrum in sample_spectra_list:
            if len(spectrum) == 0:
                padded = np.zeros(max_len)
            elif len(spectrum) < max_len:
                padded = np.pad(spectrum, (0, max_len - len(spectrum)), 'constant')
            else:
                padded = spectrum[:max_len]
            sample_spectra.append(padded)
        
        seed_spectra = np.array(seed_spectra)
        sample_spectra = np.array(sample_spectra)
        
        # Normalize spectra (avoid division by zero)
        seed_norms = np.sqrt(np.sum(seed_spectra**2, axis=1, keepdims=True))
        seed_norms[seed_norms == 0] = 1
        seed_spectra = seed_spectra / seed_norms
        
        sample_norms = np.sqrt(np.sum(sample_spectra**2, axis=1, keepdims=True))
        sample_norms[sample_norms == 0] = 1
        sample_spectra = sample_spectra / sample_norms
        
        # Calculate similarity matrix (cosine similarity * 100)
        similarity_matrix = np.dot(seed_spectra, sample_spectra.T) * 100
        
        # Get RT data
        seed_rt1 = np.array(seed_sample[0]["RT1"])
        seed_rt2 = np.array(seed_sample[0]["RT2"])
        sample_rt1 = np.array(sample[0]["RT1"])
        sample_rt2 = np.array(sample[0]["RT2"])
        
        # Calculate RT penalties - IMPORTANT: following R logic exactly
        # This creates a matrix where each column corresponds to a sample RT compared against all seed RTs
        RT1_index = np.zeros((len(seed_rt1), len(sample_rt1)))
        RT2_index = np.zeros((len(seed_rt2), len(sample_rt2)))
        
        for j, sample_rt1_val in enumerate(sample_rt1):
            RT1_index[:, j] = np.abs(sample_rt1_val - seed_rt1) * self.rt1_penalty

        for j, sample_rt2_val in enumerate(sample_rt2):
            RT2_index[:, j] = np.abs(sample_rt2_val - seed_rt2) * self.rt2_penalty
        
        # Final score = similarity - RT penalties
        return similarity_matrix - RT1_index - RT2_index


    def consensus_align_bis(self, input_file_list,
                        seed_file=0,  # Python uses 0-based indexing
                        common_ions=None,
                        # standard_library=None,
                        # nist=False
                        ):
        """
        Consensus alignment function for chromatographic data.
        
        Parameters:
        -----------
        input_file_list : list
            List of input file paths
        imported_files : list, optional
            Pre-imported files (if None, files will be imported)
        seed_file : int, default 0
            Index of the seed file (0-based indexing)
        rt1_standards : array-like, optional
            RT1 standards
        rt2_standards : array-like, optional
            RT2 standards
        c : int, default 1
            Parameter c
        rt1_penalty : int, default 1
            RT1 penalty parameter
        rt2_penalty : int, default 10
            RT2 penalty parameter
        auto_tune_match_stringency : bool, default True
            Auto-tune match stringency
        similarity_cutoff : float, default 90
            Similarity cutoff threshold
        disimilarity_cutoff : float, optional
            Dissimilarity cutoff (defaults to similarity_cutoff - 90)
        num_cores : int, default 1
            Number of cores for parallel processing
        common_ions : list, optional
            Common ions list
        missing_value_limit : float, default 0.75
            Missing value limit
        missing_peak_finder_similarity_lax : float, default 0.85
            Missing peak finder similarity lax threshold
        quant_method : str, default "T"
            Quantification method
        standard_library : optional
            Standard library
            
        Returns:
        --------
        dict : Dictionary containing alignment results with keys:
            'Alignment_Matrix', 'Peak_Info', 'RT_group', 'spectra_group'
        """
        
        # Set default values
        if self.disimilarity_cutoff is None:
            self.disimilarity_cutoff = self.similarity_cutoff - 90
        if common_ions is None:
            common_ions = []
        
       
        
        # Import files if not provided
        if self.imported_files is None:
            self.import_files(input_file_list)
        
        # Check for missing files
        missing_file_list = []
        for file_data in self.imported_files:
            if len(file_data) > 2 and file_data[2]:  # Check if there's an error message
                missing_file_list.append(file_data[2])
        
        if missing_file_list:
            raise FileNotFoundError(f"Missing files: {missing_file_list}")
        
        # Set seed sample (first file in the list by default)
        seed_sample = [df.copy() if isinstance(df, pd.DataFrame) else list(df) if isinstance(df, list) else df 
                    for df in self.imported_files[seed_file]]
        
        # Initialize matrices
        n_rows = len(seed_sample[0])
        n_cols = len(input_file_list)

        # Create row names (Name + "_1")
        row_names = [f"{seed_sample[0].iloc[i, 0]}_1" for i in range(n_rows)]

        col_names = input_file_list.copy()
        # col_names = [os.path.basename(f) for f in input_file_list]
        
        final_matrix = pd.DataFrame(np.full((n_rows, n_cols), np.nan), 
                                index=row_names, columns=col_names, dtype=float)
        final_matrix_rt = pd.DataFrame(np.full((n_rows, n_cols), None), 
                                    index=row_names, columns=col_names, dtype=object)
        final_matrix_spectra = pd.DataFrame(np.full((n_rows, n_cols), None), 
                                        index=row_names, columns=col_names, dtype=object)


        # Process each sample
        for samp_num in range(len(self.imported_files)):
            # print(f"Processing sample: {samp_num + 1}")
            # Generate similarity frames (this function needs to be implemented)
            sim_cutoffs = self.generate_sim_frames(self.imported_files[samp_num], seed_sample)

            # Afficher la valeur spécifique à seed_sample (ex. seed analyte) si tu connais son nom
            seed_name = seed_sample[0].iloc[0, 0] + "_1"  # ou autre nom exact
            # Calculate match scores (maximum similarity for each compound)
            match_scores = np.nanmax(sim_cutoffs, axis=0)
            
            # Find best matches (indices of maximum similarity)
            mates = np.nanargmax(sim_cutoffs, axis=0)
            
            # Find dissimilar matches
            dissmatch = np.where(match_scores < self.disimilarity_cutoff)[0]

            # CRITICAL: Follow R logic exactly for duplicate handling
            # 1. Create named arrays
            named_scores = {i: score for i, score in enumerate(match_scores)}
            named_mates = {i: mate for i, mate in enumerate(mates)}
            
            # 2. Sort by descending match scores
            sorted_indices = sorted(named_scores.keys(), key=lambda x: named_scores[x], reverse=True)

            # 3. Set duplicated mates to NaN
            seen_mates = set()
            for idx in sorted_indices:
                mate = named_mates[idx]
                if mate in seen_mates:
                    named_scores[idx] = np.nan
                else:
                    seen_mates.add(mate)

            # 4. Restore original order
            final_scores = np.array([named_scores[i] for i in range(len(match_scores))])
            final_mates = np.array([named_mates[i] for i in range(len(mates))])
            
            # Fill matrices based on quantification method
            if self.quant_method == "T":
                valid_matches = final_scores >= self.similarity_cutoff
                valid_indices = np.where(valid_matches)[0]
                
                for sample_idx in valid_indices:
                    seed_idx = final_mates[sample_idx]
                    if seed_idx < len(final_matrix):
                        # Fill Area (column 3 in R = index 2 in Python)
                        final_matrix.iloc[seed_idx, samp_num] = float(self.imported_files[samp_num][0].iloc[sample_idx, 2])
                        # Fill RT (column 2 in R = index 1 in Python) 
                        final_matrix_rt.iloc[seed_idx, samp_num] = str(self.imported_files[samp_num][0].iloc[sample_idx, 1])
                        # Fill Spectra (column 5 in R = index 4 in Python)
                        final_matrix_spectra.iloc[seed_idx, samp_num] = str(self.imported_files[samp_num][0].iloc[sample_idx, 4])

            # Handle dissimilar matches - add new rows
            if len(dissmatch) > 0:
                # Add to seed sample
                new_data = self.imported_files[samp_num][0].iloc[dissmatch].copy()
                seed_sample[0] = pd.concat([seed_sample[0], new_data], ignore_index=True)
                
                for dissim_idx in dissmatch:
                    seed_sample[1].append(self.imported_files[samp_num][1][dissim_idx])

                # Create new rows for matrices
                new_row_names = [f"{self.imported_files[samp_num][0].iloc[idx, 0]}_{samp_num+1}" for idx in dissmatch]

                # Create new rows filled with NaN
                new_rows_area = pd.DataFrame(np.full((len(new_row_names), len(col_names)), np.nan), 
                                            index=new_row_names, columns=col_names, dtype=float)
                new_rows_rt = pd.DataFrame(np.full((len(new_row_names), len(col_names)), None), 
                                        index=new_row_names, columns=col_names, dtype=object)
                new_rows_spectra = pd.DataFrame(np.full((len(new_row_names), len(col_names)), None), 
                                            index=new_row_names, columns=col_names, dtype=object)
                
                # Fill with current sample data
                for i, dissim_idx in enumerate(dissmatch):
                    new_rows_area.iloc[i, samp_num] = float(self.imported_files[samp_num][0].iloc[dissim_idx, 2])  # Area
                    new_rows_rt.iloc[i, samp_num] = str(self.imported_files[samp_num][0].iloc[dissim_idx, 1])      # RT
                    new_rows_spectra.iloc[i, samp_num] = str(self.imported_files[samp_num][0].iloc[dissim_idx, 4]) # Spectra
                
                # Append new rows to matrices
                final_matrix = pd.concat([final_matrix, new_rows_area])
                final_matrix_rt = pd.concat([final_matrix_rt, new_rows_rt])
                final_matrix_spectra = pd.concat([final_matrix_spectra, new_rows_spectra])


        # Update seed sample names to match final matrix row names
        if len(seed_sample[0]) != len(final_matrix):
            # Extend seed_sample[0] to match final_matrix length
            additional_rows = len(final_matrix) - len(seed_sample[0])
            if additional_rows > 0:
                # Create dummy rows
                dummy_data = pd.DataFrame({col: [np.nan] * additional_rows for col in seed_sample[0].columns})
                seed_sample[0] = pd.concat([seed_sample[0], dummy_data], ignore_index=True)
        # Update seed sample names to match final matrix row names
        seed_sample[0] = seed_sample[0].iloc[:len(final_matrix)].copy()
        seed_sample[0]['Name'] = final_matrix.index.tolist()
        
        # Order by RT (assuming RT1 is in a column called 'RT1' or similar)
        if 'RT1' in seed_sample[0].columns:
            order_rt = seed_sample[0]['RT1'].argsort()
            self.alignment_results = {
                'Alignment_Matrix': final_matrix.iloc[order_rt],
                'Peak_Info': seed_sample[0].iloc[order_rt],
                'RT_group': final_matrix_rt.iloc[order_rt],
                'spectra_group': final_matrix_spectra.iloc[order_rt],
            }
        else:
            self.alignment_results = {
                'Alignment_Matrix': final_matrix,
                'Peak_Info': seed_sample[0],
                'RT_group': final_matrix_rt,
                'spectra_group': final_matrix_spectra
            }
        
        return self.alignment_results



    def get_alignment_matrix(self):
        """Get the alignment matrix from the last alignment."""
        if self.alignment_results is None:
            raise ValueError("No alignment results available. Run consensus_align first.")
        return self.alignment_results['Alignment_Matrix']

    def get_peak_info(self):
        """Get the peak information from the last alignment."""
        if self.alignment_results is None:
            raise ValueError("No alignment results available. Run consensus_align first.")
        return self.alignment_results['Peak_Info']


    def filter_alignment_matrix(self, missing_value_threshold=None):
        """
        Filter the alignment matrix based on missing value threshold.
        
        Parameters:
        -----------
        missing_value_threshold : float, default 0.05
            Minimum proportion of non-missing values required to keep a row
            (e.g., 0.5 means keep rows with more than 50% non-missing values)
            
        Returns:
        --------
        pd.DataFrame : Filtered alignment matrix
        """ 
        if missing_value_threshold is None:
            missing_value_threshold = self.missing_value_limit
        else:
            self.missing_value_limit = missing_value_threshold

        if self.alignment_results is None:
            raise ValueError("No alignment results available. Run consensus_align first.")
    
        alignment_matrix = self.alignment_results['Alignment_Matrix'].copy()
        
        # Détermination des lignes à conserver
        non_na_count = alignment_matrix.notna().sum(axis=1)
        threshold = missing_value_threshold * alignment_matrix.shape[1]
        mask_keep = non_na_count > threshold

        print(f"Filtering: kept {mask_keep.sum()} rows out of {alignment_matrix.shape[0]} "
            f"(threshold: {missing_value_threshold*100:.0f}% non-missing values)")

        # Application du masque booléen (même position, pas besoin d'utiliser .loc avec labels)
        self.filtered_results = {
            'Alignment_Matrix': alignment_matrix[mask_keep],
            'Peak_Info': self.alignment_results['Peak_Info'].iloc[mask_keep.values].reset_index(drop=True),
            'RT_group': self.alignment_results['RT_group'].iloc[mask_keep.values],
            'spectra_group': self.alignment_results['spectra_group'].iloc[mask_keep.values]
        }


    def nist_identification(self,
                            #nist=False, 
                            match_factor_min=650):
        """
        Inject NIST identifications into Alignment_Matrix, Peak_Info, RT_group, and spectra_group.

        - Cas 1: self.alignment_results existe en mémoire (pas encore sauvegardé).
        - Cas 2: fichiers CSV sauvegardés sans identifications.
        - Cas 3: fichiers CSV sauvegardés avec identifications (re-lancement si nist=True).
        """
        if not self.nist_api.check_nist_health():
            print("⚠️ Service NIST indisponible, identifications sautées.")
            return self.filtered_results
        
        # --- CAS 1 : résultats en mémoire ---
        if self.alignment_results is not None:
            print("👉 Cas 1 : résultats en mémoire")
            spectra_group = self.filtered_results["spectra_group"]
            identifications = self._run_nist_on_spectra(spectra_group, 
                                                        #nist, 
                                                        match_factor_min)
            # Mettre à jour résultats en mémoire
            self._update_results_with_identifications(identifications)
            return self.filtered_results

        # --- CAS 2 ou 3 : résultats déjà sauvegardés ---
        elif self._csv_results_exist():
            print("👉 Cas 2/3 : résultats chargés depuis CSV")

            alignment_matrix, peak_info, rt_group, spectra_group = self._load_csv_results()

            identifications_exist = "Compounds" in peak_info.columns

            if identifications_exist: # and not nist:
                print("✅ Identifications déjà présentes, rien à faire.")
                return {
                    "Alignment_Matrix": alignment_matrix,
                    "Peak_Info": peak_info,
                    "RT_group": rt_group,
                    "spectra_group": spectra_group,
                }

            # Relancer identifications (cas 2, ou cas 3 avec nist=True)
            identifications = self._run_nist_on_spectra(spectra_group,
                                                        #True,
                                                        match_factor_min)

            # Mettre à jour les CSV
            self._update_csv_results_with_identifications(
                alignment_matrix, peak_info, rt_group, spectra_group, identifications
            )
            print("✅ Identifications NIST ajoutées aux CSV.")

            return {
                "Alignment_Matrix": alignment_matrix,
                "Peak_Info": peak_info,
                "RT_group": rt_group,
                "spectra_group": spectra_group,
            }

        else:
            raise ValueError("❌ No alignment results available. Run alignment first.")


# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------

    def _parse_spectrum_string(self, spectrum_str):
        """
        Convertit une chaîne "mz:int mz:int ..." en deux listes [mz], [int]
        """
        mz_list = []
        int_list = []
        if spectrum_str and isinstance(spectrum_str, str):
            for pair in spectrum_str.split():
                if ":" in pair:
                    mz, inten = pair.split(":")
                    mz_list.append(float(mz))
                    int_list.append(float(inten))
        # print(f"Parsed spectrum: {len(mz_list)} peaks")
        # print(int_list)
        return mz_list, int_list

    # d
    def _run_nist_on_spectra(self, spectra_group, match_factor_min=700):
        """
        Applique la recherche NIST sur chaque spectre du DataFrame spectra_group.
        Retourne un DataFrame avec les mêmes lignes/colonnes contenant 'Compounds | Scores'.
        """
        results_dict = {}

        for analyte, row in spectra_group.iterrows():
            results_dict[analyte] = {}
            for col_name, spec in row.items():
                if spec and isinstance(spec, str):  
                    try:
                        mz_list, int_list = self._parse_spectrum_string(spec)
                    except Exception as e:
                        results_dict[analyte][col_name] = f"ParseError | {e}"
                        continue

                    if mz_list and int_list:
                        serialized_spectrum = {"mass": mz_list, "intensity": int_list}
                        try:
                            results = self.nist_api.nist_single_search(serialized_spectrum)
                            list_hits = self.nist_api.hit_list_from_nist_api(results)
                            top_hits = matching.filter_best_hits(list_hits, match_factor_min)

                            if top_hits:
                                compounds = "/".join([hit[0].name for hit in top_hits])
                                scores = "/".join([str(hit[0].match_factor) for hit in top_hits])
                                results_dict[analyte][col_name] = f"{compounds} | {scores}"
                            else:
                                results_dict[analyte][col_name] = "NA"

                        except Exception as e:
                            results_dict[analyte][col_name] = f"Error | {e}"

                    else:
                        results_dict[analyte][col_name] = "NA"
                else:
                    results_dict[analyte][col_name] = "NA"

        return pd.DataFrame(results_dict).T


#     def _update_results_with_identifications(self, identifications):
#         """
#         Met à jour Alignment_Matrix, Peak_Info, RT_group et spectra_group en mémoire.
#         """
#         # Sauvegarde dans filtered_results
#         self.filtered_results["Identification"] = identifications
#         print("NIST identifications ajoutées aux résultats filtrés.")

#         # identifications est maintenant un DataFrame (comme spectra_group mais avec "compounds | scores")
#         compounds = []
#         scores = []

#         for val in identifications.stack():  # parcours de toutes les cellules
#             if isinstance(val, str) and " | " in val:
#                 comp, sc = val.split(" | ")
#                 compounds.append(comp)
#                 scores.append(sc)
#             else:
#                 compounds.append("NA")
#                 scores.append("NA")

# # Ajout dans Peak_Info
#         self.alignment_results["Peak_Info"]["Compounds"] = compounds[:len(self.alignment_results["Peak_Info"])]
#         self.alignment_results["Peak_Info"]["Scores"] = scores[:len(self.alignment_results["Peak_Info"])]

#         print("NIST identifications ajoutées à Peak_Info.")
#         # compounds = [i["Compounds"] for i in identifications]
#         # scores = [i["Scores"] for i in identifications]

#         # # Ajout dans les DataFrames en mémoire
#         # self.alignment_results["Peak_Info"]["Compounds"] = compounds
#         # self.alignment_results["Peak_Info"]["Scores"] = scores
#         # print("NIST identifications ajoutées à Peak_Info.")

#         # RT_group et spectra_group enrichis si besoin
#         self.filtered_results["RT_group"]["Compounds"] = compounds
#         print("NIST identifications ajoutées à RT_group.")
#         self.filtered_results["spectra_group"] = [
#             {**spec, "Compounds": i["Compounds"], "Scores": i["Scores"]}
#             if isinstance(spec, dict) else spec
#             for spec, i in zip(self.filtered_results["spectra_group"], identifications)
#         ]
#         print("NIST identifications ajoutées à RT_group et spectra_group.")

    # def _update_results_with_identifications(self, identifications):
    #     """
    #     Met à jour les résultats de l'alignement avec les identifications NIST.
        
    #     identifications : pd.DataFrame
    #         DataFrame renvoyé par _run_nist_on_spectra. Chaque cellule contient
    #         la chaîne "compound1/compound2 | score1/score2" ou "NA".
    #     """
    #     # Transformation en listes pour correspondre aux DataFrames existants
    #     compounds_list = []
    #     scores_list = []

    #     for row_idx, row in identifications.iterrows():
    #         for col in identifications.columns:
    #             val = row[col]
    #             if isinstance(val, str) and "|" in val:
    #                 comp_part, score_part = val.split("|", 1)
    #                 compounds_list.append(comp_part.strip())
    #                 scores_list.append(score_part.strip())
    #             else:
    #                 compounds_list.append(val)
    #                 scores_list.append("NA")

    #     # Ajustement de la longueur pour correspondre à Peak_Info
    #     n_peaks = len(self.alignment_results["Peak_Info"])
    #     self.alignment_results["Peak_Info"]["Compounds"] = compounds_list[:n_peaks]
    #     self.alignment_results["Peak_Info"]["Scores"] = scores_list[:n_peaks]

    #     # Mise à jour des RT_group si nécessaire
    #     n_rt = len(self.filtered_results["RT_group"])
    #     self.filtered_results["RT_group"]["Compounds"] = compounds_list[:n_rt]

    #     print("✅ NIST identifications mises à jour dans Peak_Info et RT_group.")


    #TODO a tester
    def _update_results_with_identifications(self, identifications):
        """
        Met à jour les DataFrames Peak_Info et RT_group avec les résultats NIST.
        identifications : DataFrame ou liste de dict avec les clés 'Compounds' et 'Scores'.
        """

        # Vérifier que Peak_Info existe
        if "Peak_Info" not in self.alignment_results:
            raise ValueError("Peak_Info non trouvé dans alignment_results")

        peak_info = self.alignment_results["Peak_Info"]
        n_rows = len(peak_info)

        # Si identifications est un DataFrame (comme retourné par _run_nist_on_spectra)
        if isinstance(identifications, pd.DataFrame):
            compounds = identifications.apply(lambda row: " | ".join(row.dropna().astype(str)), axis=1)
            scores = identifications.apply(lambda row: " | ".join(row.dropna().astype(str)), axis=1)
        # Si identifications est une liste de dict
        elif isinstance(identifications, list):
            compounds = [i.get("Compounds", "NA") for i in identifications]
            scores = [i.get("Scores", "NA") for i in identifications]
        else:
            raise TypeError("identifications doit être un DataFrame ou une liste de dict")

        # Ajuster la longueur pour correspondre à Peak_Info
        compounds = compounds[:n_rows] if len(compounds) >= n_rows else list(compounds) + ["NA"]*(n_rows - len(compounds))
        scores = scores[:n_rows] if len(scores) >= n_rows else list(scores) + ["NA"]*(n_rows - len(scores))

        # Mise à jour de Peak_Info
        peak_info["Compounds"] = compounds
        peak_info["Scores"] = scores

        # Mise à jour de RT_group si existant
        if "RT_group" in self.filtered_results:
            rt_group = self.filtered_results["RT_group"]
            rt_group["Compounds"] = compounds
            rt_group["Scores"] = scores

        print("✅ NIST identifications ajoutées à Peak_Info et RT_group.")


    def _csv_results_exist(self):
        """
        Vérifie si les CSV d'alignement existent déjà.
        """
        expected_files = [
            "alignment_matrix.csv",
            "peak_info.csv",
            "rt_group.csv",
            "spectra_group.csv",
        ]
        return all((self.output_dir / f).exists() for f in expected_files)


    def _load_csv_results(self):
        """
        Recharge les CSV d'alignement (cas 2/3).
        """
        alignment_matrix = pd.read_csv(self.output_dir / "alignment_matrix.csv", index_col=0)
        peak_info = pd.read_csv(self.output_dir / "peak_info.csv", index_col=0)
        rt_group = pd.read_csv(self.output_dir / "rt_group.csv", index_col=0)
        spectra_group = pd.read_csv(self.output_dir / "spectra_group.csv", index_col=0)
        return alignment_matrix, peak_info, rt_group, spectra_group


    def _update_csv_results_with_identifications(
        self, alignment_matrix, peak_info, rt_group, spectra_group, identifications
    ):
        """
        Met à jour les CSV avec les identifications.
        """
        compounds = [i["Compounds"] for i in identifications]
        scores = [i["Scores"] for i in identifications]

        peak_info["Compounds"] = compounds
        peak_info["Scores"] = scores

        rt_group["Compounds"] = compounds

        spectra_group["Compounds"] = compounds
        spectra_group["Scores"] = scores
        print("NIST identifications ajoutées aux DataFrames chargés depuis CSV.")

        # Sauvegarde
        alignment_matrix.to_csv(self.output_dir / "alignment_matrix.csv")
        peak_info.to_csv(self.output_dir / "peak_info.csv")
        rt_group.to_csv(self.output_dir / "rt_group.csv")
        spectra_group.to_csv(self.output_dir / "spectra_group.csv")

    # def nist_identification(self, match_factor_min=650):
    # # TODO corriger langlais
    #     """
    #     Identify the aligned spectra with NIST.
    #     # - nist = True : identifie uniquement ceux qui n ont pas encore ete identifies
    #     - nist= relance l'identification pour tous

    #     """
    #     print("NIST identification started...       ")
    #     print("nist", self.nist_api)

    #     if self.nist_api is None: #TODO a remettre
    #         raise RuntimeError("api nist is missing")
    #         # return self.filtered_results
    #     if self.alignment_results is None:
    #         raise ValueError("No alignment results available. Run the alignment first.")  

    #     if not self.nist_api.check_nist_health():#TODO a remettre
    #         print("⚠️ Service NIST indisponible, identifications sautées.")
    #         return self.filtered_results

    #     spectra_group = self.filtered_results["spectra_group"]
    #     identifications = []
        
        
    #     def parse_spectrum(spectrum):
    #         """Convertit un spectre en dict {mass, intensity}, que ce soit un objet ou une chaîne."""
    #         if spectrum is None or spectrum == "NA":
    #             return {"mass": [], "intensity": []}

    #         # Cas 1 : objet MassSpectrum, alignement en cours, avant sauvegarde ds un fichier, spectra group 
    #         if hasattr(spectrum, "mass_values"
    #             return {
    #                 "mass": [float(m) for m in spectrum.mass_values],
    #                 "intensity": [float(i) for i in spectrum.intensity_values],
    #             }

    #         # Cas 2 : dict déjà identifié
    #         if isinstance(spectrum, dict) and "Compounds" in spectrum:
    #             return spectrum  # on renvoie tel quel

    #         # Cas 3 : chaîne "mz:intensité ...", lecture a partir dun fichier .txt/.csv
    #         if isinstance(spectrum, str):
    #             try:
    #                 pairs = spectrum.strip().split()
    #                 mz, intensity = zip(*(p.split(":") for p in pairs))
    #                 return {
    #                     "mass": [float(m) for m in mz],
    #                     "intensity": [float(i) for i in intensity],
    #                 }
    #             except Exception:
    #                 return {"mass": [], "intensity": []}

    #         return {"mass": [], "intensity": []}

    #     # Parcours des spectres
    #     for spectrum in spectra_group.values.flatten():  # parcourt toutes les cellules du DataFrame
    #         already_identified = isinstance(spectrum, dict) and "Compounds" in spectrum

    #         if not already_identified: #or nist: forcer nist
    #             print("pas deja identifier , parse spectrum")
    #             serialized_spectrum = parse_spectrum(spectrum)

    #             if serialized_spectrum["mass"]:  # spectre valide
    #                 results = self.nist_api.nist_single_search(serialized_spectrum)
    #                 list_hits = self.nist_api.hit_list_from_nist_api(results)
    #                 top_hits = matching.filter_best_hits(list_hits, match_factor_min)

    #                 if top_hits:
    #                     compounds = [hit[0].name for hit in top_hits]
    #                     scores = [str(hit[0].match_factor) for hit in top_hits]
    #                     identifications.append({
    #                         "Compounds": "/".join(compounds),
    #                         "Scores": "/".join(scores)
    #                     })
    #                     print(f"Identified: {compounds} with scores {scores}")
    #                 else:
    #                     identifications.append({"Compounds": "NA", "Scores": "NA"})
    #             else:
    #                 identifications.append({"Compounds": "NA", "Scores": "NA"})
    #         else:
    #             identifications.append({
    #                 "Compounds": spectrum.get("Compounds", "NA"),
    #                 "Scores": spectrum.get("Scores", "NA")
    #             })

    #     self.filtered_results["Identification"] = identifications
    #     return self.filtered_results

    def save_results(self, output_dir, with_filter=False):
        """
        Sav) and hasattr(spectrum, "intensity_values"):e alignment results to files in tab-separated format.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        filtered_results : dict, optional
            Dictionary containing filtered versions of all matrices
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.alignment_results is None:
            raise ValueError("No alignment results available. Run consensus_align first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        name_filter = ""
        if with_filter:
            name_filter = f"_filter_{self.missing_value_limit}"
        
        # self.filtered_results['Alignment_Matrix'].to_csv(
        #     os.path.join(output_dir, f"alignment_Matrix{name_filter}_{timestamp}.csv"),
        #     sep="\t", index=True, na_rep="NA"
        # )
        # self.filtered_results['Peak_Info'].to_csv(
        #     os.path.join(output_dir, f"peak_Info{name_filter}_{timestamp}.csv"),
        #     sep="\t", index=False
        # )
        # self.filtered_results['RT_group'].to_csv(
        #     os.path.join(output_dir, f"RT_Group{name_filter}_{timestamp}.csv"),
        #     sep="\t", index=True
        # )
        # self.filtered_results['spectra_group'].to_csv(
        #     os.path.join(output_dir, f"spectra_group{name_filter}_{timestamp}.csv"),
        #     sep="\t", index=True
        # )
        # if isinstance(self.filtered_results['spectra_group'], list):
        #     try:
        #         spectra_df = pd.DataFrame(self.filtered_results['spectra_group'])
        #         spectra_df.to_csv(
        #             os.path.join(output_dir, f"spectra_group{name_filter}_{timestamp}.csv"),
        #             sep="\t", index=False
        #         )
        #     except Exception as e:
        #         print(f"❌ Impossible de sauvegarder spectra_group en CSV: {e}")
        # else:
        #     print("❌ spectra_group n'est pas une liste, impossible de créer un CSV.")

        # print(f"✅ Filter {self.missing_value_limit} applied.")
        # print(f"Results saved to directory: {output_dir}")
          # Sauvegarde des DataFrames
        for key in ['Alignment_Matrix', 'Peak_Info', 'RT_group', 'spectra_group']:
            obj = self.filtered_results.get(key)
            if obj is not None:
                try:
                    if isinstance(obj, list):
                        # convertir en DataFrame si c'est une liste
                        obj = pd.DataFrame(obj)
                    obj.to_csv(
                        os.path.join(output_dir, f"{key}{name_filter}_{timestamp}.csv"),
                        sep="\t", index=(key != 'Peak_Info'), na_rep="NA"
                    )
                    print(f"✅ {key} saved.")
                except Exception as e:
                    print(f"❌ Impossible de sauvegarder {key} en CSV: {e}")
            else:
                print(f"⚠ {key} n'existe pas dans filtered_results.")

        print(f"Results saved to directory: {output_dir}")

if __name__ == "__main__":
    # file = [
    #     "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt",
    #     "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
    #     "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt"
    # ]
    # folder = r"D:/GCxGC_MS/DATA/h5/2025_06_27_GCxGC_VOLATIL_CF_08bis_postPTR/resultPersistantHomology_tic/"
    # file = [folder + "15-04-25_817822_QC_23newE.txt",
    #     folder + "2025-04-10-854514_Q.txt",
    #     folder + "2025-04-16_751318_QCnew23E.txt",
    #     folder + "2025-05-14_817827-QC_23EI_prep22-0.txt",
    # ]

    # folder = "D:/GCxGC_MS/DATA/h5/2025-07-09_EtuVOCs_BMI_batch1bis_postPTR/result_PersistenceHomology_tic/"
    # file = [folder + "751303_v3_E3AM_5jui.txt" , 
        # folder + "751309_v3_E3PM_6jui.txt", 
        # folder + "854512_v3_E2AM_4jui.txt", 
        # folder + "854517_v3_E2AM_5jui.txt", 
        # folder + "802107_v1_E1PM_3jui.txt"
        # ]
    folder = "C:/Users/adeli/Documents/programmation/uvsq/app/output/aout_without_nist_before_align/"
    file = [folder + "A-F-028-817822-droite-ReCIV.txt",
        folder + "J-A-034-751325-Tedla.txt",
        folder + "P-L-007-801838-Tedla.txt"
        ]

    print("Importing files...", file)
    aligner = ChromatographicAligner(
        rt1_penalty=1,
        rt2_penalty=5,
        similarity_cutoff=90,
        disimilarity_cutoff=90,  # Will be set to similarity_cutoff - 90 in the function
        num_cores=1,
        missing_value_limit=0,
        quant_method="T",
        auto_tune_match_stringency=False,
        missing_peak_finder_similarity_lax=0.85

    )
    result = aligner.consensus_align_bis(
        input_file_list=file,
        seed_file=0
    )
    aligner.filter_alignment_matrix()
    aligner.nist_identification(nist=True, match_factor_min=650)
    output_dir = "C:/Users/adeli/Documents/programmation/uvsq/app/output/te1234"
    aligner.save_results(output_dir)

    # filtered_results = aligner.filter_alignment_matrix(missing_value_threshold=0.5)


    # # Save result
    # aligner.save_results(output_dir, filtered_results)