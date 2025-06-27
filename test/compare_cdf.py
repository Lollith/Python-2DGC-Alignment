import pandas as pd
file1 =
file2 =

# Lire les deux fichiers CSV
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Vérifier égalité stricte : même forme, colonnes, ordre, contenus
if df1.equals(df2):
    print("✅ Les fichiers sont identiques.")
else:
    print("❌ Les fichiers sont différents.")

    # Aide au debug :
    if df1.shape != df2.shape:
        print(f"- Forme différente : {df1.shape} vs {df2.shape}")

    if list(df1.columns) != list(df2.columns):
        print(f"- Colonnes différentes :")
        print(f"  fichier1 : {list(df1.columns)}")
        print(f"  fichier2 : {list(df2.columns)}")

    # Comparaison ligne à ligne
    diff = df1.compare(df2)
    if not diff.empty:
        print("- Différences détectées dans les valeurs :")
        print(diff)