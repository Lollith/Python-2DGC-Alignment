import pandas as pd
import numpy as np
from multiprocessing import Pool

class DebugChromatographicPrecompressFiles:
    """Version debug avec logs détaillés pour comparer avec R"""

    def __init__(self, rt1_penalty=1, rt2_penalty=10, similarity_cutoff=95, 
                 num_cores=1, common_ions=None, quant_method="T", output_files=False):
        self.rt1_penalty = rt1_penalty
        self.rt2_penalty = rt2_penalty
        self.similarity_cutoff = similarity_cutoff
        self.num_cores = num_cores
        self.common_ions = common_ions if common_ions is not None else []
        self.quant_method = quant_method
        self.output_files = output_files

    def importFile(self, file):
        print(f"\n=== IMPORTING FILE: {file} ===")
        
        # Read the file    
        current_raw_file = pd.read_csv(file, sep="\t", header=0, skipinitialspace=True)
        print(f"Initial shape: {current_raw_file.shape}")
        print(f"Initial columns: {current_raw_file.columns.tolist()}")
        print(f"First 3 rows:\n{current_raw_file.head(3)}")
        
        # Clean whitespace
        current_raw_file = current_raw_file.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

        # Convert columns to string
        current_raw_file.iloc[:, 4] = current_raw_file.iloc[:, 4].astype(str)
        current_raw_file.iloc[:, 1] = current_raw_file.iloc[:, 1].astype(str)

        # Filter out rows with NA in column 3 or empty column 5
        mask = (~pd.isna(current_raw_file.iloc[:, 2])) & (current_raw_file.iloc[:, 4].str.len() != 0) & (current_raw_file.iloc[:, 4] != 'nan')
        print(f"Rows before filtering: {len(current_raw_file)}")
        current_raw_file = current_raw_file[mask].reset_index(drop=True)
        print(f"Rows after filtering: {len(current_raw_file)}")

        # Parse retention times
        print(f"Sample RT strings: {current_raw_file.iloc[:3, 1].tolist()}")
        rt_split = current_raw_file.iloc[:, 1].str.replace('"', '', regex=False).str.split(' , ', expand=True)
        current_raw_file["RT1"] = pd.to_numeric(rt_split[0], errors='coerce')
        current_raw_file["RT2"] = pd.to_numeric(rt_split[1], errors='coerce')
        print(f"Sample RT1 values: {current_raw_file['RT1'].head(3).tolist()}")
        print(f"Sample RT2 values: {current_raw_file['RT2'].head(3).tolist()}")

        # Remove duplicates
        unique_index = (current_raw_file.iloc[:, 0].astype(str) + 
                       current_raw_file.iloc[:, 1].astype(str) + 
                       current_raw_file.iloc[:, 2].astype(str))
        
        duplicates_before = len(current_raw_file)
        current_raw_file = current_raw_file.loc[~unique_index.duplicated()].reset_index(drop=True)
        duplicates_after = len(current_raw_file)
        print(f"Duplicates removed: {duplicates_before - duplicates_after}")

        # Parse spectra data
        spectra_split = []
        ion_names = None
        print(f"Sample spectra strings: {current_raw_file.iloc[:2, 4].tolist()}")

        for i, row in current_raw_file.iterrows():
            spectrum = row.iloc[4]
            if pd.isna(spectrum) or spectrum == '' or spectrum == 'nan':
                spectra_split.append(np.array([]))
                continue
                
            peak_list = []
            for peak in str(spectrum).strip().split(" "):
                if ":" in peak:
                    parts = peak.split(":")
                    if len(parts) == 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            if mz not in self.common_ions:
                                peak_list.append((mz, intensity))
                        except ValueError:
                            continue
            
            if peak_list:
                peak_list.sort(key=lambda x: x[0])
                mzs, intensities = zip(*peak_list)
                
                if ion_names is None:
                    ion_names = list(mzs)
                    print(f"Ion names set from first spectrum: {ion_names[:5]}...")
                
                spectra_split.append(np.array(intensities))
            else:
                spectra_split.append(np.array([]))
        
        print(f"Spectra parsed: {len(spectra_split)} spectra")
        print(f"Non-empty spectra: {sum(1 for s in spectra_split if len(s) > 0)}")
        print(f"Final dataframe shape: {current_raw_file.shape}")
        
        return [current_raw_file, spectra_split, ion_names]

    def find_matches(self, sample):
        print(f"\n=== FINDING MATCHES ===")
        df, spectra_list, ion_names = sample
        print(f"Input dataframe shape: {df.shape}")
        print(f"Number of spectra: {len(spectra_list)}")
        
        if not spectra_list or len(spectra_list) == 0:
            print("No spectra to process")
            return []
        
        # Filter out empty spectra for matrix construction
        valid_indices = [i for i, s in enumerate(spectra_list) if len(s) > 0]
        if not valid_indices:
            print("No valid spectra found")
            return [[] for _ in range(len(spectra_list))]
        
        print(f"Valid spectra indices: {valid_indices[:5]}...")
        
        # Build spectra matrix only with valid spectra
        valid_spectra = [spectra_list[i] for i in valid_indices]
        spectra_matrix = np.column_stack(valid_spectra).T
        print(f"Spectra matrix shape: {spectra_matrix.shape}")
        
        # Normalize each row
        norms = np.sqrt(np.sum(spectra_matrix**2, axis=1, keepdims=True))
        norms[norms == 0] = 1
        spectra_matrix = spectra_matrix / norms
        
        # Similarity matrix
        sim_matrix = np.dot(spectra_matrix, spectra_matrix.T) * 100
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        print(f"Sample similarity values: {sim_matrix[0, :3]}")

        # RT penalties only for valid indices
        RT1_valid = df.iloc[valid_indices]['RT1'].to_numpy()
        RT2_valid = df.iloc[valid_indices]['RT2'].to_numpy()
        RT1_penalty = np.abs(RT1_valid[:, None] - RT1_valid[None, :]) * self.rt1_penalty
        RT2_penalty = np.abs(RT2_valid[:, None] - RT2_valid[None, :]) * self.rt2_penalty

        sim_matrix = sim_matrix - RT1_penalty - RT2_penalty
        np.fill_diagonal(sim_matrix, 0)
        
        print(f"Sample similarity after RT penalty: {sim_matrix[0, :3]}")
        print(f"Similarity cutoff: {self.similarity_cutoff}")

        # Create full match list (including empty spectra)
        match_list = []
        valid_idx = 0
        
        for i in range(len(spectra_list)):
            if i in valid_indices:
                matches = np.where(sim_matrix[valid_idx] >= self.similarity_cutoff)[0]
                # Convert back to original indices and make 1-based
                original_matches = [valid_indices[m] + 1 for m in matches]  # +1 for R compatibility
                match_list.append(original_matches)
                valid_idx += 1
            else:
                match_list.append([])
        
        matches_found = sum(1 for m in match_list if len(m) > 0)
        print(f"Total matches found: {matches_found}")
        print(f"Sample matches: {match_list[:3]}")
        
        return match_list
    
    def PrecompressFiles(self, input_file_list):
        print(f"\n=== STARTING PRECOMPRESS FILES ===")
        print(f"Input files: {input_file_list}")
        print(f"Parameters: RT1_penalty={self.rt1_penalty}, RT2_penalty={self.rt2_penalty}")
        print(f"Similarity cutoff: {self.similarity_cutoff}, Quant method: {self.quant_method}")
        
        combined_list = []

        # Import files (sequential for debugging)
        imported_files = []
        for file in input_file_list:
            imported_files.append(self.importFile(file))

        # Find matches (sequential for debugging)
        match_list = []
        for i, sample in enumerate(imported_files):
            print(f"\n--- Processing sample {i+1} ---")
            matches = self.find_matches(sample)
            match_list.append(matches)

        # Process each sample
        for samp_num, matches in enumerate(match_list):
            print(f"\n=== PROCESSING SAMPLE {samp_num + 1} ===")
            print(f"Matches for sample {samp_num + 1}: {matches}")
            
            if not matches or not any(len(m) > 0 for m in matches):
                print("No matches found, skipping")
                continue
                
            # Count non-empty matches
            non_empty_matches = [m for m in matches if len(m) > 0]
            if not non_empty_matches:
                print("No valid matches found")
                continue
                
            num_reps = max(len(m) for m in non_empty_matches) - 1
            print(f"Number of repetitions needed: {num_reps}")
            
            if self.quant_method in ["A", "T"]:
                # Find mates to combine
                mates = []
                for m in matches:
                    if len(m) > 0:
                        mates.append(m[0])  # First match (1-based from R)
                    else:
                        mates.append(None)
                
                print(f"Mates: {mates}")
                
                # Get valid mates (convert to 0-based for Python indexing)
                valid_mates_indices = [m-1 for m in mates if m is not None]
                rows_with_mates = [i for i, m in enumerate(mates) if m is not None]
                
                print(f"Valid mates indices (0-based): {valid_mates_indices}")
                print(f"Rows with mates: {rows_with_mates}")
                
                if valid_mates_indices:
                    df = imported_files[samp_num][0]
                    
                    # Get binding areas and create binding dataframes
                    binding_areas = df.iloc[valid_mates_indices, 2].values
                    to_bind = df.iloc[rows_with_mates, :].copy()
                    binding_partners = df.iloc[valid_mates_indices, :].copy()
                    
                    print(f"Binding areas: {binding_areas}")
                    print(f"To bind shape: {to_bind.shape}")
                    print(f"Binding partners shape: {binding_partners.shape}")
                    
                    # Create combined entry
                    combined_entry = pd.concat([
                        to_bind.reset_index(drop=True),
                        binding_partners.reset_index(drop=True),
                        pd.DataFrame({'input_file': [input_file_list[samp_num]] * len(to_bind)})
                    ], axis=1)
                    
                    print(f"Combined entry shape: {combined_entry.shape}")
                    combined_list.append(combined_entry)
                    
                    # Sum peak areas
                    to_bind.iloc[:, 2] += binding_areas
                    print(f"Updated areas: {to_bind.iloc[:, 2].values}")
                    
                    # Create Bound column and deduplicate
                    to_bind["Bound"] = [f"NA_{min(mates[i], i+1)}" for i in rows_with_mates]
                    print(f"Bound values: {to_bind['Bound'].tolist()}")
                    
                    before_dedup = len(to_bind)
                    to_bind = to_bind.drop_duplicates(subset=["Bound"])
                    after_dedup = len(to_bind)
                    print(f"Deduplication: {before_dedup} -> {after_dedup}")
                    
                    # Update imported files
                    rows_without_mates = [i for i, m in enumerate(mates) if m is None]
                    print(f"Rows without mates: {rows_without_mates}")
                    
                    imported_files[samp_num][0] = pd.concat([
                        df.iloc[rows_without_mates, :],
                        to_bind.drop(columns=["Bound"])
                    ]).reset_index(drop=True)
                    
                    print(f"Updated dataframe shape: {imported_files[samp_num][0].shape}")

        # Final combined frame
        print(f"\n=== CREATING FINAL COMBINED FRAME ===")
        print(f"Number of combined entries: {len(combined_list)}")
        
        if len(combined_list) > 0:
            for i, entry in enumerate(combined_list):
                print(f"Entry {i+1} shape: {entry.shape}")
            combined_frame = pd.concat(combined_list, ignore_index=True)
            print(f"Final combined frame shape: {combined_frame.shape}")
            print(f"Final combined frame columns: {combined_frame.columns.tolist()}")
            print(f"First few rows of combined frame:\n{combined_frame.head()}")
        else:
            combined_frame = pd.DataFrame()
            print("Empty combined frame")

        return combined_frame


# Fonction de test pour comparer étape par étape
def debug_comparison(input_files, **kwargs):
    """
    Fonction pour déboguer la comparaison avec R
    """
    debug_processor = DebugChromatographicPrecompressFiles(**kwargs)
    result = debug_processor.PrecompressFiles(input_files)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Combined frame shape: {result.shape}")
    if len(result) > 0:
        print(f"Combined frame columns: {result.columns.tolist()}")
        print(f"Sample of results:\n{result.head()}")
    
    return result

# Exemple d'utilisation:
# result = debug_comparison(['file1.txt', 'file2.txt'], 
#                          rt1_penalty=1, rt2_penalty=10, 
#                          similarity_cutoff=95, quant_method="T")

if __name__ == "__main__":
    listFiles = ["/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE.txt",
                    #   "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/2025-04-10-854514_Q.txt",
                    #   "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751310_0048GL_M1_postPTR_split.txt",
                    #   "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751315_0033CN_J7_postPTR_split.txt"
                ]
    precompressedFiles = debug_comparison(listFiles, rt1_penalty=1, rt2_penalty=10, similarity_cutoff=95, num_cores=1, common_ions=None, quant_method="T", output_files=True)
    print(precompressedFiles)