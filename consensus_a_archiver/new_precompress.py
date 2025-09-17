import pandas as pd
import numpy as np
from multiprocessing import Pool

class ChromatographicPrecompressFiles:
    """Version corrigée pour correspondre exactement au comportement R"""

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
        """Import and process chromatographic data file - exactly like R"""
        
        # Read the file    
        current_raw_file = pd.read_csv(file, sep="\t", header=0, skipinitialspace=True)
        current_raw_file = current_raw_file.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

        # Convert columns to string (columns 2 and 5 in R = indices 1 and 4 in Python)
        current_raw_file.iloc[:, 4] = current_raw_file.iloc[:, 4].astype(str)
        current_raw_file.iloc[:, 1] = current_raw_file.iloc[:, 1].astype(str)

        # Filter: !is.na(column 3) & nchar(column 5) != 0
        mask = (~pd.isna(current_raw_file.iloc[:, 2])) & (current_raw_file.iloc[:, 4].str.len() != 0) & (current_raw_file.iloc[:, 4] != 'nan')
        current_raw_file = current_raw_file[mask].reset_index(drop=True)

        # Parse retention times
        rt_split = current_raw_file.iloc[:, 1].str.replace('"', '', regex=False).str.split(' , ', expand=True)
        current_raw_file["RT1"] = pd.to_numeric(rt_split[0], errors='coerce')
        current_raw_file["RT2"] = pd.to_numeric(rt_split[1], errors='coerce')

        # Remove duplicates based on columns 1, 2, 3 (Name, RT, Area)
        unique_index = (current_raw_file.iloc[:, 0].astype(str) + 
                       current_raw_file.iloc[:, 1].astype(str) + 
                       current_raw_file.iloc[:, 2].astype(str))
        
        current_raw_file = current_raw_file.loc[~unique_index.duplicated()].reset_index(drop=True)

        # Parse spectra data
        spectra_split = []
        ion_names = None

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
                            # Exclude common ions
                            if mz not in self.common_ions:
                                peak_list.append((mz, intensity))
                        except ValueError:
                            continue
            
            if peak_list:
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
        """Calculate pairwise similarity scores - exactly like R"""
        df, spectra_list, _ = sample
        
        if not spectra_list or len(spectra_list) == 0:
            return []
        
        # Filter out empty spectra for matrix construction
        valid_indices = [i for i, s in enumerate(spectra_list) if len(s) > 0]
        if not valid_indices:
            return [[] for _ in range(len(spectra_list))]
        
        # Build spectra matrix with valid spectra only
        valid_spectra = [spectra_list[i] for i in valid_indices]
        spectra_matrix = np.column_stack(valid_spectra).T
        
        # Normalize each row
        norms = np.sqrt(np.sum(spectra_matrix**2, axis=1, keepdims=True))
        norms[norms == 0] = 1
        spectra_matrix = spectra_matrix / norms
        
        # Similarity matrix
        sim_matrix = np.dot(spectra_matrix, spectra_matrix.T) * 100

        # RT penalties only for valid indices
        RT1_valid = df.iloc[valid_indices]['RT1'].to_numpy()
        RT2_valid = df.iloc[valid_indices]['RT2'].to_numpy()
        RT1_penalty = np.abs(RT1_valid[:, None] - RT1_valid[None, :]) * self.rt1_penalty
        RT2_penalty = np.abs(RT2_valid[:, None] - RT2_valid[None, :]) * self.rt2_penalty

        sim_matrix = sim_matrix - RT1_penalty - RT2_penalty
        np.fill_diagonal(sim_matrix, 0)

        # Create full match list (including empty spectra positions)
        match_list = []
        valid_idx = 0
        
        for i in range(len(spectra_list)):
            if i in valid_indices:
                matches = np.where(sim_matrix[valid_idx] >= self.similarity_cutoff)[0]
                # Convert back to original indices and make 1-based for R compatibility
                original_matches = [valid_indices[m] + 1 for m in matches]
                match_list.append(original_matches)
                valid_idx += 1
            else:
                match_list.append([])
        
        return match_list
    
    def PrecompressFiles(self, input_file_list):
        """Main processing function - corrected to match R exactly"""
        
        # CORRECTION MAJEURE: Utiliser un dictionnaire avec les noms de fichiers comme clés
        combined_list = {}

        # Import files
        if self.num_cores > 1:
            with Pool(processes=self.num_cores) as pool:
                imported_files = pool.map(self.importFile, input_file_list)
        else:
            imported_files = [self.importFile(f) for f in input_file_list]

        # Find matches
        if self.num_cores > 1:
            with Pool(processes=self.num_cores) as pool:
                match_list = pool.map(self.find_matches, imported_files)
        else:
            match_list = [self.find_matches(sample) for sample in imported_files]

        # Process each sample
        for samp_num, matches in enumerate(match_list):
            if not matches or not any(len(m) > 0 for m in matches):
                continue
                
            num_reps = max(len(m) for m in matches if len(m) > 0) - 1
            
            if self.quant_method in ["A", "T"]:
                # Find mates to combine - PRENDRE LE PREMIER MATCH DE CHAQUE LISTE
                mates = []
                for m in matches:
                    if len(m) > 0:
                        mates.append(m[0])  # Premier match (1-based)
                    else:
                        mates.append(None)
                
                # Get indices of rows that have mates
                rows_with_mates = [i for i, m in enumerate(mates) if m is not None]
                # Get indices of the binding partners (convert to 0-based)
                valid_mates_indices = [m-1 for m in mates if m is not None]
                
                if valid_mates_indices:
                    df = imported_files[samp_num][0]
                    
                    # Get the data
                    binding_areas = df.iloc[valid_mates_indices, 2].values
                    to_bind = df.iloc[rows_with_mates, :].copy()
                    binding_partners = df.iloc[valid_mates_indices, :].copy()
                    
                    # CORRECTION: Format exact du combined_list comme en R
                    # En R: cbind(toBind, bindingPartners, inputFile)
                    combined_entry = pd.concat([
                        to_bind.reset_index(drop=True),
                        binding_partners.reset_index(drop=True),
                        pd.DataFrame({'input_file': [input_file_list[samp_num]] * len(to_bind)})
                    ], axis=1)
                    
                    # CORRECTION: Stocker avec la clé du fichier comme en R
                    if input_file_list[samp_num] not in combined_list:
                        combined_list[input_file_list[samp_num]] = combined_entry
                    else:
                        combined_list[input_file_list[samp_num]] = pd.concat([
                            combined_list[input_file_list[samp_num]], 
                            combined_entry
                        ], ignore_index=True)
                    
                    # Sum peak areas
                    to_bind.iloc[:, 2] += binding_areas
                    
                    # Create Bound column for deduplication
                    to_bind["Bound"] = [
                        f"NA_{min(mates[i], i+1)}"
                        for i in rows_with_mates
                    ]

                    to_bind = to_bind.drop_duplicates(subset=["Bound"])

                    # Update imported files - REMOVE rows with mates and ADD combined rows
                    rows_without_mates = [i for i, m in enumerate(mates) if m is None]
                    imported_files[samp_num][0] = pd.concat([
                        df.iloc[rows_without_mates, :],
                        to_bind.drop(columns=["Bound"])
                    ]).reset_index(drop=True)

            # Handle iterative combinations (num_reps > 0)
            if num_reps > 0:
                for rep in range(num_reps):
                    df = imported_files[samp_num][0].reset_index(drop=True)
                    
                    # Reparse spectra
                    spectra_split = []
                    for _, row in df.iterrows():
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
                            _, intensities = zip(*peak_list)
                            spectra_split.append(np.array(intensities))
                        else:
                            spectra_split.append(np.array([]))
                    
                    if not any(len(s) > 0 for s in spectra_split):
                        continue
                    
                    # Build and normalize spectra matrix
                    valid_indices = [i for i, s in enumerate(spectra_split) if len(s) > 0]
                    if not valid_indices:
                        continue
                        
                    valid_spectra = [spectra_split[i] for i in valid_indices]
                    spectra_matrix = np.column_stack(valid_spectra).T
                    
                    norms = np.sqrt(np.sum(spectra_matrix**2, axis=1, keepdims=True))
                    norms[norms == 0] = 1
                    spectra_matrix = spectra_matrix / norms
                    
                    similarity_matrix = (spectra_matrix @ spectra_matrix.T) * 100

                    # RT penalties
                    RT1_valid = df.iloc[valid_indices]['RT1'].to_numpy()
                    RT2_valid = df.iloc[valid_indices]['RT2'].to_numpy()
                    RT1Index = np.abs(RT1_valid[:, None] - RT1_valid[None, :]) * self.rt1_penalty
                    RT2Index = np.abs(RT2_valid[:, None] - RT2_valid[None, :]) * self.rt2_penalty
                    similarity_matrix = similarity_matrix - RT1Index - RT2Index
                    np.fill_diagonal(similarity_matrix, 0)
            
                    # Find new matches
                    new_matches = []
                    valid_idx = 0
                    for i in range(len(spectra_split)):
                        if i in valid_indices:
                            matches = np.where(similarity_matrix[valid_idx] >= self.similarity_cutoff)[0]
                            original_matches = [valid_indices[m] + 1 for m in matches]
                            new_matches.append(original_matches)
                            valid_idx += 1
                        else:
                            new_matches.append([])
                    
                    if any(len(m) > 0 for m in new_matches):
                        if self.quant_method in ["A", "T"]:
                            mates = []
                            for m in new_matches:
                                if len(m) > 0:
                                    mates.append(m[0])
                                else:
                                    mates.append(None)
                            
                            rows_with_mates = [i for i, m in enumerate(mates) if m is not None]
                            valid_mates_indices = [m-1 for m in mates if m is not None]
                            
                            if valid_mates_indices:
                                binding_areas = df.iloc[valid_mates_indices, 2].values
                                to_bind = df.iloc[rows_with_mates, :].copy()
                                binding_partners = df.iloc[valid_mates_indices, :].copy()

                                combined_entry = pd.concat([
                                    to_bind.reset_index(drop=True),
                                    binding_partners.reset_index(drop=True),
                                    pd.DataFrame({'input_file': [input_file_list[samp_num]] * len(to_bind)})
                                ], axis=1)
                                
                                # Update combined_list
                                combined_list[input_file_list[samp_num]] = pd.concat([
                                    combined_list[input_file_list[samp_num]], 
                                    combined_entry
                                ], ignore_index=True)

                                to_bind.iloc[:, 2] += binding_areas
                                to_bind["Bound"] = [
                                    f"NA_{min(mates[i], i+1)}"
                                    for i in rows_with_mates
                                ]
                                to_bind = to_bind.drop_duplicates(subset=["Bound"])

                                rows_without_mates = [i for i, m in enumerate(mates) if m is None]
                                imported_files[samp_num][0] = pd.concat([
                                    df.iloc[rows_without_mates, :],
                                    to_bind.drop(columns=["Bound"])
                                ]).reset_index(drop=True)

        # CORRECTION: Créer le combined_frame comme en R
        # En R: do.call(rbind, combinedList)
        if len(combined_list) > 0:
            combined_frame = pd.concat(list(combined_list.values()), ignore_index=True)
        else:
            combined_frame = pd.DataFrame()

        # Write output files if requested
        if self.output_files:
            for samp_num, imp in enumerate(imported_files):
                out_name = input_file_list[samp_num][:-4] + "_Py_Processed.txt"
                imp[0].iloc[:, :5].to_csv(out_name, sep="\t", index=False)

        return combined_frame

# Test function
def test_precompress(input_files, **kwargs):
    """Test function with comparison to expected R results"""
    processor = ChromatographicPrecompressFiles(**kwargs)
    result = processor.PrecompressFiles(input_files)
    
    print(f"\n=== FINAL RESULTS COMPARISON ===")
    print(f"Combined frame shape: {result.shape}")
    if len(result) > 0:
        print(f"Combined frame columns: {result.columns.tolist()}")
        print(f"Number of unique combinations: {len(result)}")
        # Show some sample combinations
        print(f"First 5 combinations:")
        for i in range(min(5, len(result))):
            print(f"  Row {i}: {result.iloc[i, 0]} + {result.iloc[i, 7]} (Areas: {result.iloc[i, 2]:.2e} + {result.iloc[i, 9]:.2e})")
    else:
        print("No combinations found")
    
    return result

if __name__ == "__main__":
    listFiles = ["/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE.txt",
                      "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/2025-04-10-854514_Q.txt",
                      "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751310_0048GL_M1_postPTR_split.txt",
                      "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751315_0033CN_J7_postPTR_split.txt"
                ]
    precompressedFiles = ChromatographicPrecompressFiles(rt1_penalty=1, rt2_penalty=10, similarity_cutoff=95, num_cores=1, common_ions=None, quant_method="T", output_files=True)
    combined_frame = precompressedFiles.PrecompressFiles(listFiles)
    print(combined_frame)