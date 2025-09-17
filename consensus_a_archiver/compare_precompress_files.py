# chemins des fichiers à comparer
# file1 = "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui_Processed.txt"
# file1 = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE_Processed.txt"
# file1 = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/2025-04-10-854514_Q_Processed.txt"
file1 = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751310_0048GL_M1_postPTR_split_ProcessedR.csv"
# file1= "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751315_0033CN_J7_postPTR_split_Processed.txt"
# file2 = "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui_Py_Processed.txt"
# file2 = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/15-04-25_817822_QC_23newE_Py_Processed.txt"
# file2 = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/2025-04-10-854514_Q_Py_Processed.txt"
file2 ="/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751310_0048GL_M1_postPTR_split_Py_Processed_last.csv"
# file2= "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/751315_0033CN_J7_postPTR_split_Py_Processed.txt"

#comparer les fichiers processed.txt
import pandas as pd
import numpy as np

df1 = pd.read_csv(file1, sep="\t")
df2 = pd.read_csv(file2, sep="\t")

# Réinitialiser les index pour que la comparaison marche ligne à ligne
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

# Vérifier si même nombre de lignes
if len(df1) != len(df2):
    print(f"⚠️ Les fichiers ont un nombre de lignes différent : {len(df1)} vs {len(df2)}")
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

cols_to_convert = ['Quant.Masses']  # ajoute d'autres colonnes si nécessaire

for col in cols_to_convert:
    # Convertir en str pour sécuriser, puis remplacer T/F par True/False
    df1[col] = df1[col].astype(str).replace({'T': True, 'F': False})
    df2[col] = df2[col].astype(str).replace({'T': True, 'F': False})
    
    # Enfin, convertir en bool
    df1[col] = df1[col].astype(bool)
    df2[col] = df2[col].astype(bool)
# Comparaison cellule par cellule
diff_df = pd.DataFrame(index=df1.index, columns=df1.columns)

for col in df1.columns:
    # Si les valeurs diffèrent, affiche "file1 | file2", sinon None
    mask = df1[col] != df2[col]
    diff_df[col] = None
    diff_df.loc[mask, col] = df1.loc[mask, col].astype(str) + " | " + df2.loc[mask, col].astype(str)

# Conserver la colonne Name telle quelle
if 'Name' in df1.columns:
    diff_df['Name'] = df1['Name']

# Sauvegarde
output_file = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/differences_full_tot1.txt"
diff_df.to_csv(output_file, sep="\t", index=False)
print(f"Différences enregistrées dans {output_file}")




# comparer les combined_frame

# Fichiers à comparer
csv_r = "combinedFrameR.csv"
csv_py = "py_combined_frame_last.csv"
output_diff = "/home/camille/Documents/app/data/cdf et h5/new/peak_detection/differences_combined_frame_tot1.csv"

# Charger les CSV
df_r = pd.read_csv(csv_r)
df_py = pd.read_csv(csv_py)

# Vérifier les colonnes
print("Colonnes R :", df_r.columns.tolist())
print("Colonnes Python :", df_py.columns.tolist())

# S'assurer qu'on compare seulement les colonnes communes
common_cols = df_r.columns.intersection(df_py.columns)
df_r = df_r[common_cols]
df_py = df_py[common_cols]

# Vérifier les dimensions
if df_r.shape != df_py.shape:
    print(f"⚠️ Dimensions différentes : R={df_r.shape}, Python={df_py.shape}")

# Comparaison ligne par ligne avec tolérance
tolerance = 1e-6
diffs = []

for i in range(min(len(df_r), len(df_py))):
    for col in common_cols:
        val_r = df_r.iloc[i][col]
        val_py = df_py.iloc[i][col]

        if pd.isna(val_r) and pd.isna(val_py):
            continue
        if isinstance(val_r, (int, float, np.number)) and isinstance(val_py, (int, float, np.number)):
            if not np.isclose(val_r, val_py, atol=tolerance):
                diffs.append({
                    "row": i,
                    "column": col,
                    "R_value": val_r,
                    "Python_value": val_py,
                    "difference": val_r - val_py if pd.notna(val_r) and pd.notna(val_py) else None
                })
        else:
            if val_r != val_py:
                diffs.append({
                    "row": i,
                    "column": col,
                    "R_value": val_r,
                    "Python_value": val_py,
                    "difference": None
                })

# Résumé
if diffs:
    diff_df = pd.DataFrame(diffs)
    diff_df.to_csv(output_diff, index=False)
    print(f"⚠️ Différences trouvées : {len(diffs)} (voir {output_diff})")
else:
    print("✅ Les CSV sont identiques (dans la tolérance définie).")