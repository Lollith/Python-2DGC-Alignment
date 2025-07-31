import numpy as np

file1_path = "d:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/Py_SimCutoffs_1.txt"
file2_path = "d:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/R_SimCutoffs_2.txt"

matrix1 = np.loadtxt(file1_path, delimiter="\t")
matrix2 = np.loadtxt(file2_path, delimiter="\t")

# Vérifier les dimensions
print("Matrix1 shape:", matrix1.shape)
print("Matrix2 shape:", matrix2.shape)

if matrix1.shape != matrix2.shape:
    print("⚠️ Les matrices n'ont pas la même taille.")
else:
    # Calcul de la différence absolue
    diff = np.abs(matrix1 - matrix2)

    # Statistiques simples
    print("📊 Différence max :", np.max(diff))
    print("📊 Différence moyenne :", np.mean(diff))
    print("📊 Nombre de valeurs différentes (> seuil) :", np.sum(diff > 1e-6))

    # Optionnel : sauvegarde des différences
    np.savetxt("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/differences.txt", diff, fmt="%.2f", delimiter="\t")
    print("✅ Fichier 'differences.txt' sauvegardé.")