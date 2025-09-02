# chemins des fichiers à comparer
file1 = "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui_Processed.txt"
file2 = "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui_Py_Processed.txt"



import pandas as pd

df1 = pd.read_csv(file1, sep="\t")
df2 = pd.read_csv(file2, sep="\t")


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
output_file = "/home/camille/Documents/app/data/output/differences_full.txt"
diff_df.to_csv(output_file, sep="\t", index=False)
print(f"Différences enregistrées dans {output_file}")