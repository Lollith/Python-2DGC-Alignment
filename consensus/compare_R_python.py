import pandas as pd
import pickle
import numpy as np

df_r = pd.read_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_current_raw_file.csv")
df_py = pd.read_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_current_raw_file.csv")

df_r[['RT1', 'RT2']] = df_r[['RT1', 'RT2']].round(6)
df_py[['RT1', 'RT2']] = df_py[['RT1', 'RT2']].round(6)

# Comparaison des dataframes
print("Dataframes identiques :", df_r.equals(df_py))
print(df_r.columns)
print(df_py.columns)
print(set(df_r.columns) - set(df_py.columns))  # colonnes dans df_r mais pas dans df_py
print(set(df_py.columns) - set(df_r.columns))  # colonnes dans df_py mais pas dans df_r
print(list(df_r.columns))
print(list(df_py.columns))



print("Colonnes identiques :", list(df_r.columns) == list(df_py.columns))
print("Shape R:", df_r.shape)
print("Shape Python:", df_py.shape)
df_r_sorted = df_r.sort_values(by=df_r.columns.tolist()).reset_index(drop=True)
df_py_sorted = df_py.sort_values(by=df_py.columns.tolist()).reset_index(drop=True)
print("Dataframes identiques (après tri) :", df_r_sorted.equals(df_py_sorted))
diff = df_r_sorted.compare(df_py_sorted)
print(diff)

diff_rt = df_r['R.T...s.'] != df_py['R.T...s.']
print(f"Differences in R.T...s.: {diff_rt.sum()} lignes")

# Voir les différences exactes
print("Exemples de différences dans R.T...s.:")
print(pd.DataFrame({
    'df_r': df_r.loc[diff_rt, 'R.T...s.'],
    'df_py': df_py.loc[diff_rt, 'R.T...s.'],
}).head(10))




# Comparaison des ions
r_ions = pd.read_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_ion_names.csv")["ion"].to_numpy()
py_ions = pd.read_csv("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_ion_names.csv")["ion"].to_numpy()

print("Ion names identiques :", np.array_equal(r_ions, py_ions))

diff_qm = df_r['Quant.Masses'] != df_py['Quant.Masses']
# print(f"Differences in Quant.Masses: {diff_qm.sum()} lignes")

df_py['Quant.Masses_clean'] = df_py['Quant.Masses'].map({'T': True, 'F': False})

# Comparer avec la colonne booléenne df_r['Quant.Masses'] (si c’est booléen)
print("Comparaison Quant.Masses :", (df_r['Quant.Masses'] == df_py['Quant.Masses_clean']).all())


# print("Exemples de différences dans Quant.Masses:")
# print(pd.DataFrame({
#     'df_r': df_r.loc[diff_qm, 'Quant.Masses'],
#     'df_py': df_py.loc[diff_qm, 'Quant.Masses'],
# }).head(10))

# Chargement des spectres
import pyreadr

# result = pyreadr.read_r("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_spectra_split.rds")
# print(result.keys())
# obj = list(result.values())[0]
# print(type(obj))
# print(obj)
# spectra_r = result[None]  # si tu veux lire le .rds depuis Python
# with open("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/py_spectra_split.pkl", "rb") as f:
#     spectra_py = pickle.load(f)

# # Optionnel : comparer un spectre
# print("Premier spectre identique :",
#       np.allclose(spectra_r[0], spectra_py[0]))
