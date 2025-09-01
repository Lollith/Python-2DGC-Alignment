# chemins des fichiers à comparer
file1 = "/home/camille/Documents/app/data/output/R_MatchList_1.txt"
file2 = "/home/camille/Documents/app/data/output/py_MatchList_1.txt"

# lire les fichiers
with open(file1, 'r') as f1:
    lines1 = [line.strip() for line in f1]

with open(file2, 'r') as f2:
    lines2 = [line.strip() for line in f2]

# comparer ligne par ligne
max_len = max(len(lines1), len(lines2))
for i in range(max_len):
    l1 = lines1[i] if i < len(lines1) else "<no line>"
    l2 = lines2[i] if i < len(lines2) else "<no line>"
    if l1 != l2:
        print(f"Ligne {i+1} diffère:")
        print(f"  Fichier 1: {l1}")
        print(f"  Fichier 2: {l2}")
