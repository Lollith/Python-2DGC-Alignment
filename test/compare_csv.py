import pandas as pd
import numpy as np
from pathlib import Path

def compare_csv_files(file1_path, file2_path, output_path=None):
    """
    Compare deux fichiers CSV ligne par ligne
    
    Args:
        file1_path (str): Chemin vers le premier fichier CSV
        file2_path (str): Chemin vers le deuxième fichier CSV  
        output_path (str, optional): Chemin pour sauvegarder le rapport
    
    Returns:
        dict: Résumé des différences
    """
    
    print(f"📊 Comparaison de {file1_path} et {file2_path}")
    
    try:
        # Charger les fichiers CSV
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        print(f"📄 Fichier 1: {len(df1)} lignes, {len(df1.columns)} colonnes")
        print(f"📄 Fichier 2: {len(df2)} lignes, {len(df2.columns)} colonnes")
        
        results = {
            'identical_rows': 0,
            'different_rows': 0,
            'differences': [],
            'missing_in_file1': [],
            'missing_in_file2': []
        }
        
        # Vérifier si les colonnes sont identiques
        if list(df1.columns) != list(df2.columns):
            print("⚠️ ATTENTION: Les colonnes ne sont pas identiques!")
            print(f"Colonnes fichier 1: {list(df1.columns)}")
            print(f"Colonnes fichier 2: {list(df2.columns)}")
        
        # Comparer ligne par ligne
        max_rows = max(len(df1), len(df2))
        
        for i in range(max_rows):
            # Ligne manquante dans fichier 1
            if i >= len(df1):
                results['missing_in_file1'].append({
                    'row_index': i,
                    'content': df2.iloc[i].to_dict()
                })
                continue
            
            # Ligne manquante dans fichier 2
            if i >= len(df2):
                results['missing_in_file2'].append({
                    'row_index': i,
                    'content': df1.iloc[i].to_dict()
                })
                continue
            
            # Comparer les lignes existantes
            row1 = df1.iloc[i]
            row2 = df2.iloc[i]
            
            # Vérifier si les lignes sont identiques
            if row1.equals(row2):
                results['identical_rows'] += 1
            else:
                results['different_rows'] += 1
                
                # Identifier les colonnes différentes
                diff_cols = {}
                for col in df1.columns:
                    if col in df2.columns:
                        val1 = row1[col]
                        val2 = row2[col]
                        
                        # Gérer les valeurs NaN
                        if pd.isna(val1) and pd.isna(val2):
                            continue
                        elif val1 != val2:
                            diff_cols[col] = {
                                'file1': val1,
                                'file2': val2
                            }
                
                if diff_cols:
                    results['differences'].append({
                        'row_index': i,
                        'differences': diff_cols
                    })
        
        # Afficher le résumé
        print(f"\n📈 RÉSUMÉ:")
        print(f"✅ Lignes identiques: {results['identical_rows']}")
        print(f"❌ Lignes différentes: {results['different_rows']}")
        print(f"📭 Lignes manquantes dans fichier 1: {len(results['missing_in_file1'])}")
        print(f"📭 Lignes manquantes dans fichier 2: {len(results['missing_in_file2'])}")
        
        # Afficher quelques exemples de différences
        if results['differences']:
            print(f"\n🔍 Exemples de différences (5 premières):")
            for i, diff in enumerate(results['differences'][:5]):
                print(f"  Ligne {diff['row_index']}:")
                for col, values in diff['differences'].items():
                    print(f"    {col}: '{values['file1']}' ≠ '{values['file2']}'")
        
        # Sauvegarder le rapport si demandé
        if output_path:
            save_comparison_report(results, df1, df2, output_path)
            print(f"💾 Rapport sauvegardé: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def save_comparison_report(results, df1, df2, output_path):
    """Sauvegarder un rapport détaillé des différences"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE COMPARAISON CSV\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Lignes identiques: {results['identical_rows']}\n")
        f.write(f"Lignes différentes: {results['different_rows']}\n")
        f.write(f"Lignes manquantes dans fichier 1: {len(results['missing_in_file1'])}\n")
        f.write(f"Lignes manquantes dans fichier 2: {len(results['missing_in_file2'])}\n\n")
        
        # Détail des différences
        if results['differences']:
            f.write("DÉTAIL DES DIFFÉRENCES:\n")
            f.write("-" * 30 + "\n")
            
            for diff in results['differences']:
                f.write(f"\nLigne {diff['row_index']}:\n")
                for col, values in diff['differences'].items():
                    f.write(f"  {col}:\n")
                    f.write(f"    Fichier 1: {values['file1']}\n")
                    f.write(f"    Fichier 2: {values['file2']}\n")

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplace par tes chemins de fichiers
    file1 = "/home/camille/Documents/app/data/output/output_align/new/spectra_group_20250917_120536.csv"
    file2 = "/home/camille/Documents/app/data/output/output_align/spectra_group_20250828_150258.csv"
    rapport = "rapport_comparaison.txt"
    
    results = compare_csv_files(file1, file2, rapport)
