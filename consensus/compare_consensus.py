# # import pandas as pd
# # import numpy as np
# # import os

# # output_dir = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/"

# # def compare_matrices(file_py, file_r, name, index_col=0):
# #     print(f"\n🔍 Comparing: {name}")
# #     #

# #     # df_py = pd.read_csv(os.path.join(output_dir, file_py), sep="\t", index_col=index_col, na_values="NA")
# #     # df_r = pd.read_csv(os.path.join(output_dir, file_r), sep="\t", index_col=index_col, na_values="NA")

# #     df_py = pd.read_csv(os.path.join(output_dir, file_py), sep="\t", index_col=index_col, na_values=["", "NA", '""'])
# #     df_r  = pd.read_csv(os.path.join(output_dir, file_r), sep="\t", index_col=index_col, na_values=["", "NA", '""'])



# #     # Dimensions
# #     print(f"  - Python shape: {df_py.shape}")
# #     print(f"  - R shape     : {df_r.shape}")

# #     # Vérifier lignes et colonnes
# #     if not df_py.index.equals(df_r.index):
# #         print("  ❌ Index mismatch.")
# #     else:
# #         print("  ✅ Index match.")

# #     if not df_py.columns.equals(df_r.columns):
# #         print("  ❌ Columns mismatch.")
# #     else:
# #         print("  ✅ Columns match.")

# #     # Comparaison cellule à cellule
# #     diffs = []
# #     for row in df_py.index:
# #         for col in df_py.columns:
# #             val_py = df_py.at[row, col]
# #             val_r = df_r.at[row, col]

# #             if pd.isna(val_py) and pd.isna(val_r):
# #                 continue
# #             elif pd.isna(val_py) or pd.isna(val_r):
# #                 diffs.append((row, col, val_py, val_r, "NA mismatch"))
# #             else:
# #                 try:
# #                     # Convertir en float si possible
# #                     val_py_float = float(val_py)
# #                     val_r_float = float(val_r)
# #                     if not np.isclose(val_py_float, val_r_float, rtol=1e-03, atol=1e-02):
# #                         diffs.append((row, col, val_py, val_r, f"diff = {abs(val_py_float - val_r_float):.4f}"))
# #                 except:
# #                     if str(val_py) != str(val_r):
# #                         diffs.append((row, col, val_py, val_r, "string mismatch"))

# #     print(f"  ➤ {len(diffs)} differences found.")
# #     if diffs:
# #         print("  First differences:")
# #         for d in diffs[:10]:
# #             print(f"    Row: {d[0]} | Col: {d[1]} | Py: {d[2]} | R: {d[3]} | {d[4]}")
# #     return diffs

# # # Comparaison des matrices après filtrage
# # diff1 = compare_matrices("py_Alignment_Matrix_after_filter.txt", "R_Alignment_Matrix_after_filter.txt", "Alignment Matrix (filtered)")

# # # Comparaison des matrices RT group
# # diff2 = compare_matrices("py_RT_Group.txt", "R_RT_Group.txt", "RT Group")

# # # Comparaison des matrices spectra
# # diff3 = compare_matrices("py_Spectra_Group.txt", "R_Spectra_Group.txt", "Spectra Group")

# # # Comparaison des Peak Info : ici, pas d’index
# # diff4 = compare_matrices("py_Peak_Info.txt", "R_Peak_Info.txt", "Peak Info", index_col=None)

# # # Résumé global
# # total_diffs = sum(len(d) for d in [diff1, diff2, diff3, diff4])
# # print(f"\n✅ Comparaison terminée : {total_diffs} différences détectées au total.")
# import pandas as pd
# import numpy as np
# import os

# output_dir = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/"
# file_py = "py_peak_info.txt"
# file_r = "R_Peak_Info.txt"
# peak_info_py = pd.read_csv(os.path.join(output_dir, file_py), sep="\t", index_col=0, na_values="NA")
# peak_info_r = pd.read_csv(os.path.join(output_dir, file_r), sep="\t", index_col=0, na_values="NA")

# def normalize_bool_str(v):
#     if isinstance(v, str):
#         if v == "T":
#             return "True"
#         elif v == "F":
#             return "False"
#     return v

# def parse_rt(rt_str):
#     if pd.isna(rt_str):
#         return []
#     parts = [p.strip() for p in str(rt_str).split(',')]
#     try:
#         return [float(p) for p in parts]
#     except:
#         return parts

# def parse_spectra(spectra_str):
#     if pd.isna(spectra_str):
#         return []
#     pairs = spectra_str.strip().split()
#     result = []
#     for pair in pairs:
#         try:
#             mz, intensity = pair.split(':')
#             result.append((float(mz), float(intensity)))
#         except:
#             result.append(pair)
#     return result

# def normalize_cell(value):
#     """Nettoie les cellules pour permettre une comparaison plus souple."""
#     # Remplacer 'T' / 'F' par True / False
#     if isinstance(value, str):
#         val = value.strip()
#         if val == 'T':
#             return True
#         elif val == 'F':
#             return False
#         elif val == '':
#             return np.nan
#         else:
#             return val
#     return value

# def compare_matrices(df1, df2, name=""):
#     print(f"🔍 Comparing: {name}")
#     print(f"  - Python shape: {df1.shape}")
#     print(f"  - R shape     : {df2.shape}")

#     # Vérifier colonnes et index
#     if not df1.columns.equals(df2.columns):
#         print("  ❌ Columns mismatch.")
#     else:
#         print("  ✅ Columns match.")

#     if not df1.index.equals(df2.index):
#         print("  ❌ Index mismatch.")
#     else:
#         print("  ✅ Index match.")

#     # Appliquer la normalisation
#     df1_clean = df1.map(normalize_cell)
#     df2_clean = df2.map(normalize_cell)

#     # Remplir les NaN pour éviter les erreurs dans les comparaisons
#     df1_clean = df1_clean.fillna("MISSING")
#     df2_clean = df2_clean.fillna("MISSING")

#     # Comparaison élément par élément
#     differences = []
#     for i in df1_clean.index:
#         for col in df1_clean.columns:
#             val1 = df1_clean.at[i, col]
#             val2 = df2_clean.at[i, col]

#             if isinstance(val1, float) and isinstance(val2, float):
#                 if not np.isclose(val1, val2, atol=1e-6):
#                     differences.append((i, col, val1, val2))
#             else:
#                 if val1 != val2:
#                     differences.append((i, col, val1, val2))

#     print(f"  ➤ {len(differences)} differences found.")
#     if differences:
#         print("  First differences:")
#         for i, (row, col, v1, v2) in enumerate(differences[:10]):
#             print(f"    Row: {row} | Col: {col} | Py: {v1} | R: {v2} | string mismatch")

#     return differences

# compare_matrices(peak_info_py, peak_info_r, name="Peak Info")


import pandas as pd
import numpy as np
import os

output_dir = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/"

def normalize_cell(value):
    """Nettoie les cellules pour permettre une comparaison plus souple."""
    if isinstance(value, str):
        val = value.strip().strip('"')
        if val in ['T', 'True']: return True
        if val in ['F', 'False']: return False
        if val == "": return np.nan
        return val
    return value

def compare_matrices(file_py, file_r, name, index_col=0):
    print(f"\n🔍 Comparing: {name}")

    df_py = pd.read_csv(os.path.join(output_dir, file_py), sep="\t", index_col=index_col, na_values=["", "NA", '""'])
    df_r  = pd.read_csv(os.path.join(output_dir, file_r), sep="\t", index_col=index_col, na_values=["", "NA", '""'])
    df_py.columns = df_py.columns.str.strip()
    df_r.columns = df_r.columns.str.strip()


    print(f"  - Python shape: {df_py.shape}")
    print(f"  - R shape     : {df_r.shape}")

    # Vérifier index et colonnes
    if not df_py.index.equals(df_r.index):
        print("  ❌ Index mismatch.")
    else:
        print("  ✅ Index match.")

    if not df_py.columns.equals(df_r.columns):
        print("  ❌ Columns mismatch.")
    else:
        print("  ✅ Columns match.")

    # Appliquer normalisation
    df1 = df_py.map(normalize_cell)
    df2 = df_r.map(normalize_cell)

    # Remplir les NaN
    df1 = df1.fillna("MISSING")
    df2 = df2.fillna("MISSING")

    # Comparaison cellule par cellule
    diffs = []
    for row in df1.index:
        for col in df1.columns:
            val1 = df1.at[row, col]
            val2 = df2.at[row, col]

            if isinstance(val1, float) and isinstance(val2, float):
                if not np.isclose(val1, val2, rtol=1e-3, atol=1e-6):
                    diffs.append((row, col, val1, val2, f"float diff = {abs(val1 - val2):.6f}"))
            elif val1 != val2:
                diffs.append((row, col, val1, val2, "string mismatch"))

    print(f"  ➤ {len(diffs)} differences found.")
    if diffs:
        print("  First differences:")
        for d in diffs[:10]:
            print(f"    Row: {d[0]} | Col: {d[1]} | Py: {d[2]} | R: {d[3]} | {d[4]}")
    return diffs

# Comparaison des fichiers
diff1 = compare_matrices("py_Alignment_Matrix_after_filter.txt", "R_Alignment_Matrix_after_filter.txt", "Alignment Matrix (filtered)")
diff2 = compare_matrices("py_RT_Group.txt", "R_RT_Group.txt", "RT Group")
diff3 = compare_matrices("py_Spectra_Group.txt", "R_Spectra_Group.txt", "Spectra Group")
diff4 = compare_matrices("py_Peak_Info.txt", "R_Peak_Info.txt", "Peak Info", index_col=0)

# Résumé global
total_diffs = sum(len(d) for d in [diff1, diff2, diff3, diff4])
print(f"\n✅ Comparaison terminée : {total_diffs} différences détectées au total.")
