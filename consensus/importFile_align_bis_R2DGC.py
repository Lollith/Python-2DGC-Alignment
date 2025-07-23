import pandas as pd
import numpy as np

def importFile(file):
    missing_standards = []

    #read the file    
    current_raw_file = pd.read_csv(file, sep="\t", header=0,skipinitialspace=True)
    current_raw_file = current_raw_file.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print(current_raw_file.iloc[:, 1].head())

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


if __name__ == "__main__":

    #DEBUG
    file = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/A-F-028-817822-droite-ReCIV.txt"
    with open(file, "r", encoding="utf-8") as f:
        line = f.readlines()
        print(len(line))
        print(line[584])



    # importFile("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/A-F-028-817822-droite-ReCIV.txt")
    df, spectra_split, missing_standards, ion_names, spectra_split = importFile("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/A-F-028-817822-droite-ReCIV.txt")
    df.to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_current_raw_file.csv", index=False)

    import pickle
    with open("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_spectra_split.pkl", "wb") as f:
        pickle.dump(spectra_split, f)   

    pd.DataFrame({"ion": ion_names}).to_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_ion_names.csv", index=False)