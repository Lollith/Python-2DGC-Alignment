import pandas as pd
import numpy as np

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


# def generate_sim_frames(sample, seed_sample, RT2Penalty=5, RT1Penalty=1):
#     # Extraction des m/z (optionnel selon usage)
#     mz_seed = seed_sample[3]
#     mz_sample = sample[3]
#     print(f"Seed m/z: {mz_seed}, Sample m/z: {mz_sample}")
#     print(f"Are m/z equal? {mz_seed == mz_sample}")

#     # Création de la matrice des spectres (chaque ligne = un pic)
#     # seed_spectra = np.array(seed_sample[1]).T
#     # seed_spectra = seed_spectra / np.sqrt((seed_spectra**2).sum(axis=1, keepdims=True))

#     # sample_spectra = np.array(sample[1]).T
#     # sample_spectra = sample_spectra / np.sqrt((sample_spectra**2).sum(axis=1, keepdims=True))
#     seed_spectra = np.array([s.flatten() for s in seed_sample[1]])
#     sample_spectra = np.array([s.flatten() for s in sample[1]])
#     seed_spectra = seed_spectra / np.linalg.norm(seed_spectra, axis=1, keepdims=True)
#     sample_spectra = sample_spectra / np.linalg.norm(sample_spectra, axis=1, keepdims=True)
#     print(f"Seed spectra shape: {seed_spectra.shape}")

#     print(f"Sample spectra shape: {sample_spectra.shape}")

#     # Calcul de la similarité cosinus entre tous les pics des deux échantillons
#     similarity_matrix = np.dot(seed_spectra, sample_spectra.T) * 100

#     # Calcul des pénalités de rétention (RT1 et RT2)
#     seed_rt1 = np.array(seed_sample[0]["RT1"])
#     sample_rt1 = np.array(sample[0]["RT1"])

#     seed_rt2 = np.array(seed_sample[0]["RT2"])
#     sample_rt2 = np.array(sample[0]["RT2"])

#     RT1_index = np.abs(seed_rt1[:, None] - sample_rt1[None, :]) * RT1Penalty
#     RT2_index = np.abs(seed_rt2[:, None] - sample_rt2[None, :]) * RT2Penalty


#     # Résultat final = score de similarité - pénalité de RT1 - pénalité de RT2
#     return similarity_matrix - RT1_index - RT2_index


# def generate_sim_frames(sample, seed_sample, RT2Penalty=5, RT1Penalty=1):
#     # Extraction des m/z (optionnel selon usage)
#     mz_seed = seed_sample[3]
#     mz_sample = sample[3]
#     # print(f"Seed m/z: {mz_seed}, Sample m/z: {mz_sample}")
#     # print(f"Are m/z equal? {mz_seed == mz_sample}")

#     # Création de la matrice des spectres (chaque ligne = un pic)
#     # seed_spectra = np.array(seed_sample[1]).T
#     # seed_spectra = seed_spectra / np.sqrt((seed_spectra**2).sum(axis=1, keepdims=True))

#     # sample_spectra = np.array(sample[1]).T
#     # sample_spectra = sample_spectra / np.sqrt((sample_spectra**2).sum(axis=1, keepdims=True))
#     seed_spectra = np.array([s.flatten() for s in seed_sample[1]])
#     sample_spectra = np.array([s.flatten() for s in sample[1]])
#     seed_spectra = seed_spectra / np.linalg.norm(seed_spectra, axis=1, keepdims=True)
#     sample_spectra = sample_spectra / np.linalg.norm(sample_spectra, axis=1, keepdims=True)
#     print(f"Seed spectra shape: {seed_spectra.shape}")

#     print(f"Sample spectra shape: {sample_spectra.shape}")

#     # Calcul de la similarité cosinus entre tous les pics des deux échantillons
#     similarity_matrix = np.dot(seed_spectra, sample_spectra.T) * 100

#     # Calcul des pénalités de rétention (RT1 et RT2)
#     # seed_rt1 = np.array(seed_sample[0]["RT1"])
#     # sample_rt1 = np.array(sample[0]["RT1"])

#     # seed_rt2 = np.array(seed_sample[0]["RT2"])
#     # sample_rt2 = np.array(sample[0]["RT2"])
#     seed_rt1 = np.array(seed_sample[0]["RT1"])[:seed_spectra.shape[0]]
#     sample_rt1 = np.array(sample[0]["RT1"])[:sample_spectra.shape[0]]
#     seed_rt2 = np.array(seed_sample[0]["RT2"])[:seed_spectra.shape[0]]
#     sample_rt2 = np.array(sample[0]["RT2"])[:sample_spectra.shape[0]]

#     print("seed_rt1 length:", len(seed_rt1))
#     print("seed_spectra shape:", seed_spectra.shape)
#     print("sample_rt1 length:", len(sample_rt1))
#     print("sample_spectra shape:", sample_spectra.shape)

#     # RT1_index = np.abs(seed_rt1[:, None] - sample_rt1[None, :]) * RT1Penalty
#     # RT2_index = np.abs(seed_rt2[:, None] - sample_rt2[None, :]) * RT2Penalty
#        # Construire RT1_index en itérant sur sample_rt1 et comparant avec seed_rt1, 
#     # puis assembler le tout en matrice avec nrow = similarity_matrix.shape[0]
#     RT1_index_rows = []
#     for rt1 in sample_rt1:
#         diff = np.abs(rt1 - seed_rt1) * RT1Penalty
#         RT1_index_rows.append(diff)
#     RT1_index = np.vstack(RT1_index_rows).T  # transpose pour matcher shape (seed_peaks, sample_peaks)

#     RT2_index_rows = []
#     for rt2 in sample_rt2:
#         diff = np.abs(rt2 - seed_rt2) * RT2Penalty
#         RT2_index_rows.append(diff)
#     RT2_index = np.vstack(RT2_index_rows).T

#     print("similarity_matrix shape:", similarity_matrix.shape)
#     print("RT1_index shape:", RT1_index.shape)
#     print("RT2_index shape:", RT2_index.shape)
#     # Résultat final = score de similarité - pénalité de RT1 - pénalité de RT2
#     return similarity_matrix - RT1_index - RT2_index

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
    
# def generate_sim_frames(sample, seed_sample, rt2_penalty=5, rt1_penalty=1):
#     """
#     Réplication exacte de la fonction R GenerateSimFrames
    
#     Parameters:
#     -----------
#     sample : list
#         Liste contenant [RT_data, spectra_list, ..., mz_data]
#     seed_sample : list  
#         Liste contenant [RT_data, spectra_list, ..., mz_data]
#     rt2_penalty : float, default=5
#     rt1_penalty : float, default=1
        
#     Returns:
#     --------
#     numpy.ndarray
#         Matrice de similarité identique à la sortie R
#     """
    
#     # Extraire mz (4ème élément en R, index 3 en Python)
#     mz_seed = seed_sample[3]
#     mz_sample = sample[3]
    
#     print(f"mz_seed: {mz_seed}, mz_sample: {mz_sample}")
#     print(f"mz_seed == mz_sample: {mz_seed == mz_sample}")
    
#     # === TRAITEMENT DES SPECTRES SEED ===
#     # Équivalent exact de: do.call(cbind, SeedSample[[2]])
#     seed_spectra_list = seed_sample[1]  # 2ème élément en R = index 1 en Python
    
#     # Créer la matrice en combinant les colonnes (cbind)
#     if isinstance(seed_spectra_list, list):
#         seed_spectra_frame = np.column_stack(seed_spectra_list)
#     else:
#         seed_spectra_frame = seed_spectra_list
    
#     # Transposition: t(seedSpectraFrame)
#     seed_spectra_frame = seed_spectra_frame.T
    
#     # Normalisation exacte comme en R:
#     # as.matrix(seedSpectraFrame)/sqrt(apply((as.matrix(seedSpectraFrame))^2, 1, sum))
#     seed_spectra_matrix = seed_spectra_frame.astype(float)
    
#     # Calcul des normes par ligne (axis=1)
#     row_sums_squared = np.sum(seed_spectra_matrix**2, axis=1)
#     row_norms = np.sqrt(row_sums_squared)
    
#     # Éviter division par zéro (comme R le fait implicitement)
#     row_norms[row_norms == 0] = 1
    
#     # Normalisation par ligne
#     seed_spectra_frame = seed_spectra_matrix / row_norms[:, np.newaxis]
    
#     print(f"Seed spectra shape: {seed_spectra_frame.shape}")
    
#     # === TRAITEMENT DES SPECTRES SAMPLE ===
#     sample_spectra_list = sample[1]
    
#     if isinstance(sample_spectra_list, list):
#         sample_spectra_frame = np.column_stack(sample_spectra_list)
#     else:
#         sample_spectra_frame = sample_spectra_list
        
#     sample_spectra_frame = sample_spectra_frame.T
#     sample_spectra_matrix = sample_spectra_frame.astype(float)
    
#     row_sums_squared = np.sum(sample_spectra_matrix**2, axis=1)
#     row_norms = np.sqrt(row_sums_squared)
#     row_norms[row_norms == 0] = 1
    
#     sample_spectra_frame = sample_spectra_matrix / row_norms[:, np.newaxis]
    
#     print(f"Sample spectra shape: {sample_spectra_frame.shape}")
    
#     # === CALCUL DE LA MATRICE DE SIMILARITÉ ===
#     # Équivalent exact de: (seedSpectraFrame %*% t(sampleSpectraFrame)) * 100
#     similarity_matrix = np.dot(seed_spectra_frame, sample_spectra_frame.T) * 100
    
#     print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
#     # === CALCUL DES PÉNALITÉS RT ===
    
#     # Extraire les données RT (1er élément en R = index 0 en Python)
#     sample_rt_data = sample[0]
#     seed_rt_data = seed_sample[0]
    
#     # Gestion flexible des formats de données RT
#     if isinstance(sample_rt_data, pd.DataFrame):
#         sample_rt1 = sample_rt_data['RT1'].values
#         sample_rt2 = sample_rt_data['RT2'].values
#         seed_rt1 = seed_rt_data['RT1'].values  
#         seed_rt2 = seed_rt_data['RT2'].values
#     elif hasattr(sample_rt_data, 'dtype') and hasattr(sample_rt_data.dtype, 'names'):
#         # Array numpy structuré
#         sample_rt1 = sample_rt_data['RT1']
#         sample_rt2 = sample_rt_data['RT2'] 
#         seed_rt1 = seed_rt_data['RT1']
#         seed_rt2 = seed_rt_data['RT2']
#     else:
#         # Array numpy simple - supposer que RT1 est colonne 0, RT2 colonne 1
#         sample_rt1 = sample_rt_data[:, 0]
#         sample_rt2 = sample_rt_data[:, 1]
#         seed_rt1 = seed_rt_data[:, 0]
#         seed_rt2 = seed_rt_data[:, 1]
    
#     print(f"RT1 shapes - Sample: {sample_rt1.shape}, Seed: {seed_rt1.shape}")
#     print(f"RT2 shapes - Sample: {sample_rt2.shape}, Seed: {seed_rt2.shape}")
    
#     # === CALCUL RT1 INDEX ===
#     # Réplication exacte de:
#     # matrix(unlist(lapply(Sample[[1]][, "RT1"], 
#     #        function(x) abs(x - SeedSample[[1]][, "RT1"]) * RT1Penalty)),
#     #        nrow = nrow(SimilarityMatrix))
    
#     # Pour chaque RT1 du sample, calculer la différence avec tous les RT1 du seed
#     rt1_differences = []
#     for sample_rt1_val in sample_rt1:
#         diff_vector = np.abs(sample_rt1_val - seed_rt1) * rt1_penalty
#         rt1_differences.extend(diff_vector)
    
#     # Convertir en matrice avec les bonnes dimensions
#     # nrow = nrow(SimilarityMatrix) = nombre de lignes de seed_spectra
#     rt1_index = np.array(rt1_differences).reshape(similarity_matrix.shape, order='F')  # 'F' pour ordre colonne-major comme R
    
#     # === CALCUL RT2 INDEX ===
#     rt2_differences = []
#     for sample_rt2_val in sample_rt2:
#         diff_vector = np.abs(sample_rt2_val - seed_rt2) * rt2_penalty
#         rt2_differences.extend(diff_vector)
    
#     rt2_index = np.array(rt2_differences).reshape(similarity_matrix.shape, order='F')
    
#     print(f"RT1 index shape: {rt1_index.shape}")
#     print(f"RT2 index shape: {rt2_index.shape}")
    
#     # === RÉSULTAT FINAL ===
#     result = similarity_matrix - rt1_index - rt2_index
    
#     print(f"Final result shape: {result.shape}")
#     print(f"Result range: [{result.min():.2f}, {result.max():.2f}]")
    
#     return result

if __name__ == "__main__":
    listFiles = ["D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt",
                 "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
                 "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt"]
    ImportedFiles = [importFile(file) for file in listFiles]    
    # seed est le 1er fichier de la liste
    SeedSample = ImportedFiles[0]
    print("SeedSample:", SeedSample[3])
    # Génération des frames de similarité pour chaque échantillon
    for SampNum in range(1, len(ImportedFiles)):
        if SampNum != 0:
            # print("file", ImportedFiles[SampNum] )
            SimCutoffs = generate_sim_frames(ImportedFiles[SampNum], SeedSample)
            print(SimCutoffs)
            np.savetxt(f"D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/Py_SimCutoffs_{SampNum}.txt", SimCutoffs, delimiter="\t")