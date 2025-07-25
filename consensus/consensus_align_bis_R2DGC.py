import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import warnings
import os
import platform


def importFile(file):
    missing_standards = []

    #read the file    
    current_raw_file = pd.read_csv(file, sep="\t", header=0,skipinitialspace=True)
    current_raw_file = current_raw_file.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # print(current_raw_file.iloc[:, 1].head())

    # Convert columns to string
    current_raw_file.iloc[:, 4] = current_raw_file.iloc[:, 4].astype(str)
    current_raw_file.iloc[:, 1] = current_raw_file.iloc[:, 1].astype(str)

    # rajoute les ""
    current_raw_file['R.T...s.'] = '"' + current_raw_file['R.T...s.'].str.strip('"') + '"'
    
    #split RTs
    current_raw_file[["RT1", "RT2"]] = current_raw_file["R.T...s."].str.replace('"', '', regex=False).str.split(" , ", expand=True).astype(float)

    # suppression des doublons
    composite_key = composite_key = current_raw_file["Name"].astype(str) + "_" + \
                current_raw_file["R.T...s."].astype(str) + "_" + \
                current_raw_file["Area"].astype(str)
    current_raw_file = current_raw_file.loc[~composite_key.duplicated()].reset_index(drop=True)

    # Spectres en liste par ligne (chaque spectre est une liste d'intensités)
    spectra_split = []
    ion_names = None
    for i, row in current_raw_file.iterrows():
        spectrum = row.iloc[4]
        peak_list = [p.split(":") for p in spectrum.strip().split(" ") if ":" in p]
        try:
            mzs, intensities = zip(*[(float(mz), float(intensity)) for mz, intensity in peak_list])
        except ValueError:
            mzs, intensities = [], []

        # On trie par m/z croissant
        sorted_pairs = sorted(zip(mzs, intensities), key=lambda x: x[0])
        sorted_mzs, sorted_intensities = zip(*sorted_pairs) if sorted_pairs else ([], [])
        if ion_names is None and sorted_mzs:
            ion_names = list(sorted_mzs)
        
        spectra_split.append(np.array(sorted_intensities))

    return [current_raw_file, spectra_split, missing_standards, ion_names, spectra_split]


def generate_sim_frames(sample, seed_sample, RT2Penalty=5, RT1Penalty=1):
    # Extraction des m/z (optionnel selon usage)
    mz_seed = seed_sample[3]
    mz_sample = sample[3]
    print(f"Seed m/z: {mz_seed}, Sample m/z: {mz_sample}")
    print(f"Are m/z equal? {mz_seed == mz_sample}")

    # Création de la matrice des spectres (chaque ligne = un pic)
    # seed_spectra = np.array(seed_sample[1]).T
    # seed_spectra = seed_spectra / np.sqrt((seed_spectra**2).sum(axis=1, keepdims=True))

    # sample_spectra = np.array(sample[1]).T
    # sample_spectra = sample_spectra / np.sqrt((sample_spectra**2).sum(axis=1, keepdims=True))
    seed_spectra = np.array([s.flatten() for s in seed_sample[1]])
    sample_spectra = np.array([s.flatten() for s in sample[1]])
    seed_spectra = seed_spectra / np.linalg.norm(seed_spectra, axis=1, keepdims=True)
    sample_spectra = sample_spectra / np.linalg.norm(sample_spectra, axis=1, keepdims=True)
    print(f"Seed spectra shape: {seed_spectra.shape}")

    print(f"Sample spectra shape: {sample_spectra.shape}")

    # Calcul de la similarité cosinus entre tous les pics des deux échantillons
    similarity_matrix = np.dot(seed_spectra, sample_spectra.T) * 100

    # Calcul des pénalités de rétention (RT1 et RT2)
    seed_rt1 = np.array(seed_sample[0]["RT1"])
    sample_rt1 = np.array(sample[0]["RT1"])

    seed_rt2 = np.array(seed_sample[0]["RT2"])
    sample_rt2 = np.array(sample[0]["RT2"])

    RT1_index = np.abs(seed_rt1[:, None] - sample_rt1[None, :]) * RT1Penalty
    RT2_index = np.abs(seed_rt2[:, None] - sample_rt2[None, :]) * RT2Penalty


    # Résultat final = score de similarité - pénalité de RT1 - pénalité de RT2
    return similarity_matrix - RT1_index - RT2_index

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
    seed_sample = imported_files[seed_file].copy()
    
    # Initialize matrices
    n_rows = len(seed_sample[0])
    n_cols = len(input_file_list)
    
    final_matrix = np.full((n_rows, n_cols), np.nan)
    final_matrix_rt = np.full((n_rows, n_cols), np.nan)
    final_matrix_spectra = np.full((n_rows, n_cols), np.nan)
    
    # Create row and column names
    row_names = [f"{seed_sample[0].iloc[i, 0]}_1" for i in range(n_rows)]
    col_names = input_file_list.copy()
    
    # Convert matrices to DataFrames for easier indexing
    final_matrix = pd.DataFrame(final_matrix, index=row_names, columns=col_names)
    final_matrix_rt = pd.DataFrame(final_matrix_rt, index=row_names, columns=col_names)
    final_matrix_spectra = pd.DataFrame(final_matrix_spectra, index=row_names, columns=col_names)
    
    # Process each sample
    for samp_num in range(len(imported_files)):
        # Generate similarity frames (this function needs to be implemented)
        sim_cutoffs = generate_sim_frames(imported_files[samp_num], seed_sample, rt2_penalty, rt1_penalty)
        
        # Calculate match scores (maximum similarity for each compound)
        match_scores = np.nanmax(sim_cutoffs, axis=0)
        
        # Find best matches (indices of maximum similarity)
        mates = np.nanargmax(sim_cutoffs, axis=0)
        
        # Find dissimilar matches
        dissmatch = np.where(match_scores < disimilarity_cutoff)[0]
        
        # Sort by match scores (descending)
        sorted_indices = np.argsort(-match_scores)
        sorted_mates = mates[sorted_indices]
        sorted_scores = match_scores[sorted_indices]
        
        # Handle duplicates - set duplicated mates to NaN
        _, unique_indices = np.unique(sorted_mates, return_index=True)
        duplicate_mask = np.ones(len(sorted_mates), dtype=bool)
        duplicate_mask[unique_indices] = False
        sorted_scores[duplicate_mask] = np.nan
        
        # Restore original order
        restore_order = np.argsort(sorted_indices)
        mates = sorted_mates[restore_order]
        match_scores = sorted_scores[restore_order]
        
        # Fill matrices based on quantification method
        if quant_method == "T":
            valid_matches = match_scores >= similarity_cutoff
            valid_indices = np.where(valid_matches)[0]
            
            if len(valid_indices) > 0:
                for idx in valid_indices:
                    mate_idx = mates[idx]
                    if mate_idx < len(final_matrix):
                        # Area values
                        final_matrix.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 2]  # Area column
                        # RT values  
                        final_matrix_rt.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 1]  # RT column
                        # Spectra values
                        final_matrix_spectra.iloc[mate_idx, samp_num] = imported_files[samp_num][0].iloc[idx, 3]  # Spectra column
        
        # Handle dissimilar matches - add new rows
        if len(dissmatch) > 0:
            # Add to seed sample
            new_data = imported_files[samp_num][0].iloc[dissmatch].copy()
            seed_sample[0] = pd.concat([seed_sample[0], new_data], ignore_index=True)
            
            # Update seed_sample[1] if it exists (assuming it's a dictionary or list)
            if len(seed_sample) > 1 and isinstance(seed_sample[1], dict):
                start_idx = len(seed_sample[1])
                for i, dissim_idx in enumerate(dissmatch):
                    seed_sample[1][str(start_idx + i + 1)] = imported_files[samp_num][1][dissim_idx]
            
            # Create new rows for matrices
            # n_new_rows = len(dissmatch)
            new_row_names = [f"{imported_files[samp_num][0].iloc[idx, 0]}_{samp_num+1}" for idx in dissmatch]
            
            # Create new rows filled with NaN
            new_rows_area = pd.DataFrame(np.nan, index=new_row_names, columns=col_names)
            new_rows_rt = pd.DataFrame(np.nan, index=new_row_names, columns=col_names)
            new_rows_spectra = pd.DataFrame(np.nan, index=new_row_names, columns=col_names)
            
            # Fill with current sample data
            for i, dissim_idx in enumerate(dissmatch):
                new_rows_area.iloc[i, samp_num] = imported_files[samp_num][0].iloc[dissim_idx, 2]  # Area
                new_rows_rt.iloc[i, samp_num] = imported_files[samp_num][0].iloc[dissim_idx, 1]    # RT
                new_rows_spectra.iloc[i, samp_num] = imported_files[samp_num][0].iloc[dissim_idx, 3]  # Spectra
            
            # Append new rows to matrices
            final_matrix = pd.concat([final_matrix, new_rows_area])
            final_matrix_rt = pd.concat([final_matrix_rt, new_rows_rt])
            final_matrix_spectra = pd.concat([final_matrix_spectra, new_rows_spectra])
    
    # Update seed sample names
    seed_sample[0]['Name'] = final_matrix.index.tolist()
    
    # Order by RT (assuming RT1 is in a column called 'RT1' or similar)
    if 'RT1' in seed_sample[0].columns:
        order_rt = seed_sample[0]['RT1'].argsort()
    elif len(seed_sample[0].columns) > 1:
        # Assume RT is in the second column
        order_rt = seed_sample[0].iloc[:, 1].argsort()
    else:
        order_rt = np.arange(len(seed_sample[0]))
    
    # Create return dictionary
    return_dict = {
        'Alignment_Matrix': final_matrix.iloc[order_rt],
        'Peak_Info': seed_sample[0].iloc[order_rt],
        'RT_group': final_matrix_rt.iloc[order_rt],
        'spectra_group': final_matrix_spectra.iloc[order_rt]
    }
    
    return return_dict


if __name__ == "__main__":
    file = [
        "C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751303_v3_E3AM_5jui.txt",
        "C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751304_v1_E3AM_4jui.txt",
        "C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751306_v1_E3PM_5jui.txt"
    ]
    alignment = consensus_align_bis(
        input_file_list=file,
        seed_file=1,
        missing_value_limit=0,
        rt2_penalty=5,
        rt1_penalty=1,
        similarity_cutoff=90,
        disimilarity_cutoff=None,
        num_cores=cpu_count(),
        common_ions=None
    )

    alignment_filtered_matrix = alignment['Alignment_Matrix'].copy()
    my_filter = 0.5
    indexkeep = alignment_filtered_matrix.isna().mean(axis=1) < my_filter

    alignment_filtered_matrix = alignment_filtered_matrix[indexkeep]
    alignment_filtered_matrix.to_csv("C:/Users/adeli/Documents/programmation/uvsq/Python-2DGC-Alignment/consensus/py_alignment_matrix_after_filter.txt", sep="\t", index=True)

    alignment["Peak_Info"].to_csv("C:/Users/adeli/Documents/programmation/uvsq/Python-2DGC-Alignment/consensus/py_peak_info.txt", sep="\t", index=True)
    alignment["RT_group"].to_csv("C:/Users/adeli/Documents/programmation/uvsq/Python-2DGC-Alignment/consensus/py_rt_group.txt", sep="\t", index=True)
    alignment["spectra_group"].to_csv("C:/Users/adeli/Documents/programmation/uvsq/Python-2DGC-Alignment/consensus/py_spectra_group.txt", sep="\t", index=True)