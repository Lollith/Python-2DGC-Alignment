import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import warnings
import os
import platform


def importFile(file):
    """Import and process chromatographic data file"""
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

    # Spectres en liste par ligne (chaque spectre est une liste d'intensités)
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
    
def generate_sim_frames(sample, seed_sample, RT2Penalty=5, RT1Penalty=1):
    """Generate similarity frames between sample and seed"""
    
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
    # In R: matrix(unlist(lapply(Sample[[1]][, "RT1"], function(x) abs(x - SeedSample[[1]][, "RT1"]) * RT1Penalty))
    # This creates a matrix where each column corresponds to a sample RT compared against all seed RTs
    
    RT1_index = np.zeros((len(seed_rt1), len(sample_rt1)))
    RT2_index = np.zeros((len(seed_rt2), len(sample_rt2)))
    
    for j, sample_rt1_val in enumerate(sample_rt1):
        RT1_index[:, j] = np.abs(sample_rt1_val - seed_rt1) * RT1Penalty
    
    for j, sample_rt2_val in enumerate(sample_rt2):
        RT2_index[:, j] = np.abs(sample_rt2_val - seed_rt2) * RT2Penalty
    
    # Final score = similarity - RT penalties
    return similarity_matrix - RT1_index - RT2_index

# def generate_sim_frames(sample, seed_sample, RT2Penalty=10, RT1Penalty=1):
#     """Generate similarity frames between sample and seed"""
    
#     # Get spectra data
#     seed_spectra_list = seed_sample[1]
#     sample_spectra_list = sample[1]
    
#     # Find maximum spectrum length
#     all_spectra = [s for s in seed_spectra_list + sample_spectra_list if len(s) > 0]
#     if not all_spectra:
#         max_len = 1
#     else:
#         max_len = max(len(s) for s in all_spectra)
    
#     # Pad spectra to same length
#     def pad_spectrum(spectrum, target_len):
#         if len(spectrum) == 0:
#             return np.zeros(target_len)
#         elif len(spectrum) < target_len:
#             return np.pad(spectrum, (0, target_len - len(spectrum)), 'constant')
#         else:
#             return spectrum[:target_len]
    
#     seed_spectra = np.array([pad_spectrum(s, max_len) for s in seed_spectra_list])
#     sample_spectra = np.array([pad_spectrum(s, max_len) for s in sample_spectra_list])
    
#     # Normalize spectra (avoid division by zero)
#     def normalize_spectra(spectra):
#         norms = np.sqrt(np.sum(spectra**2, axis=1, keepdims=True))
#         norms[norms == 0] = 1
#         return spectra / norms
    
#     seed_spectra = normalize_spectra(seed_spectra)
#     sample_spectra = normalize_spectra(sample_spectra)
    
#     # Calculate similarity matrix (cosine similarity * 100)
#     similarity_matrix = np.dot(seed_spectra, sample_spectra.T) * 100
    
#     # Get RT data
#     seed_rt1 = np.array(seed_sample[0]["RT1"])
#     seed_rt2 = np.array(seed_sample[0]["RT2"])  
#     sample_rt1 = np.array(sample[0]["RT1"])
#     sample_rt2 = np.array(sample[0]["RT2"])
    
#     # Calculate RT penalties following R logic exactly
#     # R creates matrix where each column j represents sample j compared to all seed peaks
#     RT1_index = np.zeros((len(seed_rt1), len(sample_rt1)))
#     RT2_index = np.zeros((len(seed_rt2), len(sample_rt2)))
    
#     for j, sample_rt1_val in enumerate(sample_rt1):
#         RT1_index[:, j] = np.abs(sample_rt1_val - seed_rt1) * RT1Penalty
    
#     for j, sample_rt2_val in enumerate(sample_rt2):
#         RT2_index[:, j] = np.abs(sample_rt2_val - seed_rt2) * RT2Penalty
    
#     # Final score = similarity - RT penalties
#     return similarity_matrix - RT1_index - RT2_index


def consensus_align_bis(input_file_list,
                    #    imported_files=None,
                       seed_file=0,  # Python uses 0-based indexing
                       rt1_standards=None,
                       rt2_standards=None,
                       c=1,
                       rt1_penalty=1,
                       rt2_penalty=10,
                       auto_tune_match_stringency=True,
                       similarity_cutoff=90,
                       disimilarity_cutoff=None,
                       num_cores=1,
                       common_ions=None,
                       missing_value_limit=0.75,
                       missing_peak_finder_similarity_lax=0.85,
                       quant_method="T",
                       standard_library=None):
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
    if disimilarity_cutoff is None:
        disimilarity_cutoff = similarity_cutoff - 90
    if common_ions is None:
        common_ions = []
    
    # Import files if not provided
    # if imported_files is None:
    if num_cores == 1:
        imported_files = [importFile(file) for file in input_file_list]
    else:
        # Ensure compatibility with both Windows and Linux
        # On Windows, use 'spawn' method for multiprocessing
        if platform.system() == 'Windows':
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            imported_files = list(executor.map(importFile, input_file_list))
    
    # Check for missing files
    missing_file_list = []
    for file_data in imported_files:
        if len(file_data) > 2 and file_data[2]:  # Check if there's an error message
            missing_file_list.append(file_data[2])
    
    if missing_file_list:
        raise FileNotFoundError(f"Missing files: {missing_file_list}")
    
    # Set seed sample (first file in the list by default)
    # seed_sample = imported_files[seed_file].copy()
      # Set seed sample (deep copy to avoid modifying original)
    seed_sample = [df.copy() if isinstance(df, pd.DataFrame) else list(df) if isinstance(df, list) else df 
                   for df in imported_files[seed_file]]
    
    # Initialize matrices
    n_rows = len(seed_sample[0])
    n_cols = len(input_file_list)

     # Create row names (Name + "_1")
    row_names = [f"{seed_sample[0].iloc[i, 0]}_1" for i in range(n_rows)]
    col_names = input_file_list.copy()
    
    # final_matrix = np.full((n_rows, n_cols), np.nan)
    # final_matrix_rt = np.full((n_rows, n_cols), np.nan)
    # final_matrix_spectra = np.full((n_rows, n_cols), np.nan)
    final_matrix = pd.DataFrame(np.full((n_rows, n_cols), np.nan), 
                               index=row_names, columns=col_names, dtype=float)
    final_matrix_rt = pd.DataFrame(np.full((n_rows, n_cols), None), 
                                  index=row_names, columns=col_names, dtype=object)
    final_matrix_spectra = pd.DataFrame(np.full((n_rows, n_cols), None), 
                                       index=row_names, columns=col_names, dtype=object)
    
    # # Create row and column names
    # row_names = [f"{seed_sample[0].iloc[i, 0]}_1" for i in range(n_rows)]
    # col_names = input_file_list.copy()
    
    # Convert matrices to DataFrames for easier indexing
    # final_matrix = pd.DataFrame(final_matrix, index=row_names, columns=col_names)
    # # final_matrix_rt = pd.DataFrame(final_matrix_rt, index=row_names, columns=col_names, dtype=object)
    # # final_matrix_spectra = pd.DataFrame(final_matrix_spectra, index=row_names, columns=col_names, dtype=object)
    # final_matrix_rt = pd.DataFrame(final_matrix_rt, index=row_names, columns=col_names)
    # final_matrix_spectra = pd.DataFrame(final_matrix_spectra, index=row_names, columns=col_names)

    # Process each sample
    for samp_num in range(len(imported_files)):
        print(f"Processing sample: {samp_num + 1}")
        # Generate similarity frames (this function needs to be implemented)
        sim_cutoffs = generate_sim_frames(imported_files[samp_num], seed_sample, rt2_penalty, rt1_penalty)
        print(f"Similarity matrix shape: {sim_cutoffs.shape}")
        # Calculate match scores (maximum similarity for each compound)
        match_scores = np.nanmax(sim_cutoffs, axis=0)
        
        # Find best matches (indices of maximum similarity)
        mates = np.nanargmax(sim_cutoffs, axis=0)
        
        print(f"Match scores shape: {match_scores.shape}")
        print(f"Mates shape: {mates.shape}")
        print(f"Number of valid matches (>= {similarity_cutoff}): {np.sum(match_scores >= similarity_cutoff)}")
        # Find dissimilar matches
        dissmatch = np.where(match_scores < disimilarity_cutoff)[0]
        print(f"Number of dissimilar matches: {len(dissmatch)}")

        # # Sort by match scores (descending)
        # order_indices = np.argsort(-match_scores)
        # sorted_mates = mates[order_indices]
        # sorted_scores = match_scores[order_indices]
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
         # Handle duplicates - set scores of duplicated mates to NaN
        # seen_mates = set()
        # for i, mate in enumerate(sorted_mates):
        #     if mate in seen_mates:
        #         sorted_scores[i] = np.nan
        #     else:
        #         seen_mates.add(mate)

        # # Handle duplicates - set duplicated mates to NaN
        # _, unique_indices = np.unique(sorted_mates, return_index=True)
        # duplicate_mask = np.ones(len(sorted_mates), dtype=bool)
        # duplicate_mask[unique_indices] = False
        # sorted_scores[duplicate_mask] = np.nan
        
        # Restore original order
        # restore_order = np.argsort(order_indices)
        # final_mates = sorted_mates[restore_order]
        # final_scores = sorted_scores[restore_order]
        
        # Fill matrices based on quantification method
        if quant_method == "T":
            valid_matches = final_scores >= similarity_cutoff
            valid_indices = np.where(valid_matches)[0]
            print(f"Final valid matches: {len(valid_indices)}")
            # if len(valid_indices) > 0:
            #     for idx in valid_indices:
            #         mate_idx = mates[idx]
            #         if mate_idx < len(final_matrix):
            #             # Area values
            #             final_matrix.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 2]  # Area column
            #             # RT values  
            #             final_matrix_rt.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 1]  # RT column
            #             # Spectra values
            #             final_matrix_spectra.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 3]  # Spectra column
            for sample_idx in valid_indices:
                seed_idx = final_mates[sample_idx]
                if seed_idx < len(final_matrix):
                    # Fill Area (column 3 in R = index 2 in Python)
                    final_matrix.iloc[seed_idx, samp_num] = float(imported_files[samp_num][0].iloc[sample_idx, 2])
                    # Fill RT (column 2 in R = index 1 in Python) 
                    final_matrix_rt.iloc[seed_idx, samp_num] = str(imported_files[samp_num][0].iloc[sample_idx, 1])
                    # Fill Spectra (column 5 in R = index 4 in Python)
                    final_matrix_spectra.iloc[seed_idx, samp_num] = str(imported_files[samp_num][0].iloc[sample_idx, 4])

        # Handle dissimilar matches - add new rows
        if len(dissmatch) > 0:
            # Add to seed sample
            print(f"Adding {len(dissmatch)} new rows for dissimilar matches")
            new_data = imported_files[samp_num][0].iloc[dissmatch].copy()
            seed_sample[0] = pd.concat([seed_sample[0], new_data], ignore_index=True)
            
            # Update seed_sample[1] if it exists (assuming it's a dictionary or list)
            # if len(seed_sample) > 1 and isinstance(seed_sample[1], dict):
            #     start_idx = len(seed_sample[1])
                # for i, dissim_idx in enumerate(dissmatch):
                    # seed_sample[1][str(start_idx + i + 1)] = imported_files[samp_num][1][dissim_idx]
              # Add to seed spectra
            for dissim_idx in dissmatch:
                seed_sample[1].append(imported_files[samp_num][1][dissim_idx])
            
            # Create new rows for matrices
            # n_new_rows = len(dissmatch)
            new_row_names = [f"{imported_files[samp_num][0].iloc[idx, 0]}_{samp_num+1}" for idx in dissmatch]
            
            # Create new rows filled with NaN
            new_rows_area = pd.DataFrame(np.full((len(new_row_names), len(col_names)), np.nan), 
                                        index=new_row_names, columns=col_names, dtype=float)
            new_rows_rt = pd.DataFrame(np.full((len(new_row_names), len(col_names)), None), 
                                      index=new_row_names, columns=col_names, dtype=object)
            new_rows_spectra = pd.DataFrame(np.full((len(new_row_names), len(col_names)), None), 
                                           index=new_row_names, columns=col_names, dtype=object)
            
            # Fill with current sample data
            for i, dissim_idx in enumerate(dissmatch):
                new_rows_area.iloc[i, samp_num] = float(imported_files[samp_num][0].iloc[dissim_idx, 2])  # Area
                new_rows_rt.iloc[i, samp_num] = str(imported_files[samp_num][0].iloc[dissim_idx, 1])      # RT
                new_rows_spectra.iloc[i, samp_num] = str(imported_files[samp_num][0].iloc[dissim_idx, 4]) # Spectra
            
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
    # Update seed sample names
    # seed_sample[0]['Name'] = final_matrix.index.tolist()
    # Update seed sample names to match final matrix row names
    seed_sample[0] = seed_sample[0].iloc[:len(final_matrix)].copy()
    seed_sample[0]['Name'] = final_matrix.index.tolist()
    
    # Order by RT (assuming RT1 is in a column called 'RT1' or similar)
    if 'RT1' in seed_sample[0].columns:
        order_rt = seed_sample[0]['RT1'].argsort()
        return_dict = {
            'Alignment_Matrix': final_matrix.iloc[order_rt],
            'Peak_Info': seed_sample[0].iloc[order_rt],
            'RT_group': final_matrix_rt.iloc[order_rt],
            'spectra_group': final_matrix_spectra.iloc[order_rt]
        }
    else:
        return_dict = {
            'Alignment_Matrix': final_matrix,
            'Peak_Info': seed_sample[0],
            'RT_group': final_matrix_rt,
            'spectra_group': final_matrix_spectra
        }
    # Create return dictionary
    # return_dict = {
    #     'Alignment_Matrix': final_matrix.iloc[order_rt],
    #     'Peak_Info': seed_sample[0].iloc[order_rt],
    #     'RT_group': final_matrix_rt.iloc[order_rt],
    #     'spectra_group': final_matrix_spectra.iloc[order_rt]
    # }
    
    return return_dict


# if __name__ == "__main__":
#     file = [
#         "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt",
#         "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
#         "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt"
#     ]
#     num_cores = min(os.cpu_count(), 60)
#     alignment = consensus_align_bis(
#         input_file_list=file,
#         seed_file=0,
#         missing_value_limit=0,
#         rt2_penalty=5,
#         rt1_penalty=1,
#         similarity_cutoff=90,
#         disimilarity_cutoff=90,
#         num_cores=num_cores,
#         common_ions=None
#     )
#     # print(alignment_matrix)


#     alignment_filtered_matrix = alignment['Alignment_Matrix'].copy()
#     my_filter = 0.5
#     indexkeep = alignment_filtered_matrix.isna().mean(axis=1) < my_filter

#     alignment_filtered_matrix = alignment_filtered_matrix[indexkeep]
#     alignment_filtered_matrix.to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_alignment_matrix_after_filter.txt", sep="\t", index=True)

#     alignment["Peak_Info"].to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_peak_info.txt", sep="\t", index=True)
#     alignment["RT_group"].to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_rt_group.txt", sep="\t", index=True)
#     alignment["spectra_group"].to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_spectra_group.txt", sep="\t", index=True)
if __name__ == "__main__":
    # Test files
    file = [
        "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt",
        "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
        "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt"
    ]
    
    # Run alignment with same parameters as R
    alignment = consensus_align_bis(
        input_file_list=file,
        seed_file=0,
        missing_value_limit=0,
        rt2_penalty=5,
        rt1_penalty=1,
        similarity_cutoff=90,
        disimilarity_cutoff=90,  # Same as R: similarityCutoff - 90 + 90
        num_cores=1,  # Use 1 core for debugging
        auto_tune_match_stringency=False,
        missing_peak_finder_similarity_lax=0.85,
        quant_method="T"
    )
    
    print("Alignment Matrix shape:", alignment['Alignment_Matrix'].shape)
    print("\nFirst few rows of Alignment Matrix:")
    print(alignment['Alignment_Matrix'].head(15))
    
    # Apply filter
    alignment_filtered_matrix = alignment['Alignment_Matrix'].copy()
    my_filter = 0.5
    # Keep rows where more than 50% are not missing (same as R logic)
    non_na_count = alignment_filtered_matrix.notna().sum(axis=1)
    threshold = my_filter * alignment_filtered_matrix.shape[1]
    index_keep = non_na_count > threshold
    
    alignment_filtered_matrix = alignment_filtered_matrix[index_keep]
    
    # Save results
    output_dir = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/"
    
    alignment['Alignment_Matrix'].to_csv(
        os.path.join(output_dir, "py_Alignment_Matrix.txt"), 
        sep="\t", index=True, na_rep="NA"
    )
    
    alignment['Peak_Info'].to_csv(
        os.path.join(output_dir, "py_Peak_Info.txt"), 
        sep="\t", index=False
    )
    
    alignment['RT_group'].to_csv(
        os.path.join(output_dir, "py_RT_Group.txt"), 
        sep="\t", index=True
    )
    
    alignment['spectra_group'].to_csv(
        os.path.join(output_dir, "py_Spectra_Group.txt"), 
        sep="\t", index=True
    )
    
    alignment_filtered_matrix.to_csv(
        os.path.join(output_dir, "py_Alignment_Matrix_after_filter.txt"), 
        sep="\t", index=True, na_rep="NA"
    )
    
    print(f"\nFiltered matrix shape: {alignment_filtered_matrix.shape}")
    print("Results saved successfully!")