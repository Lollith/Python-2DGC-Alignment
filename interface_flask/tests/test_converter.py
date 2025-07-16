
import os
import h5py
import pytest
import shutil
import numpy as np
from data_converter import DataConverter

# Configuration des chemins
TEST_DATA_DIR = "tests/tests_data"
EXPECTED_DIR = "tests/reference_results"
OUTPUT_DIR = "tests/output_tmp"

FILES = [
    "A-F-028-817822-droite-ReCIVA.cdf",
    "J-A-034-751325-Tedlar.cdf"
]

@pytest.fixture
def converter():
    """Fixture pour créer une instance de DataConverter."""
    return DataConverter()

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Fixture pour nettoyer avant et après chaque test."""
    # Setup: créer le répertoire de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Nettoyer les fichiers existants
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    yield  # Exécuter le test
    

def compare_hdf5_files(expected_file, result_file, tolerance=1e-6):
    """
    Compare deux fichiers HDF5 avec une tolérance pour les valeurs numériques.
    
    Args:
        expected_file: Chemin vers le fichier de référence
        result_file: Chemin vers le fichier généré
        tolerance: Tolérance pour la comparaison des valeurs float
    """
    with h5py.File(expected_file, "r") as expected, h5py.File(result_file, "r") as result:
        # Vérifier que les datasets sont les mêmes
        expected_keys = set(expected.keys())
        result_keys = set(result.keys())
        
        assert expected_keys == result_keys, f"Mismatch in datasets. Expected: {expected_keys}, Got: {result_keys}"
        
        # Comparer chaque dataset
        for key in expected_keys:
            expected_data = expected[key][:]
            result_data = result[key][:]
            
            # Vérifier les formes
            assert expected_data.shape == result_data.shape, f"Shape mismatch for {key}: expected {expected_data.shape}, got {result_data.shape}"
            
            # Comparer selon le type de données
            if expected_data.dtype.kind in ['f', 'c']:  # float ou complex
                assert np.allclose(expected_data, result_data, rtol=tolerance, atol=tolerance), f"Values mismatch in dataset: {key}"
            else:  # integer, boolean, string
                assert np.array_equal(expected_data, result_data), f"Values mismatch in dataset: {key}"
        
        # Comparer les attributs si nécessaire
        for attr_name in expected.attrs:
            if attr_name in result.attrs:
                assert expected.attrs[attr_name] == result.attrs[attr_name], f"Attribute mismatch: {attr_name}"

@pytest.mark.parametrize("filename", FILES)
def test_cdf_conversion(converter, filename):
    """Test de conversion CDF vers HDF5."""
    input_path = TEST_DATA_DIR
    output_path = OUTPUT_DIR
    
    # Vérifier que le fichier d'entrée existe
    input_file = os.path.join(input_path, filename)
    assert os.path.exists(input_file), f"Input file not found: {input_file}"
    
    # Effectuer la conversion
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        input_path,
        [filename],
        output_path
    )
    
    # Vérifier que la conversion a réussi
    assert success, f"Conversion failed for {filename}. Messages: {messages}"
    assert len(converted_files) == 1, f"Expected 1 converted file, got {len(converted_files)}"
    
    # Vérifier que le fichier de sortie existe
    output_file = converted_files[0]
    assert os.path.exists(output_file), f"Output file not created: {output_file}"
    
    # Vérifier l'extension du fichier de sortie
    assert output_file.endswith('.h5'), f"Output file should be .h5, got: {output_file}"
    
    # Chemin du fichier attendu
    expected_filename = filename.replace('.cdf', '.h5')
    expected_file = os.path.join(EXPECTED_DIR, expected_filename)
    
    if os.path.exists(expected_file):
        # Comparaison des fichiers HDF5
        compare_hdf5_files(expected_file, output_file)
        print(f"✅ Comparison successful for {filename}")
    else:
        print(f"⚠️  Expected file not found: {expected_file}")
        print(f"   Generated file: {output_file}")
        print("   Consider copying the generated file to expected directory if it's correct")

def test_batch_conversion(converter):
    """Test de conversion en lot."""
    input_path = TEST_DATA_DIR
    output_path = OUTPUT_DIR
    
    # Vérifier que les fichiers d'entrée existent
    existing_files = []
    for filename in FILES:
        input_file = os.path.join(input_path, filename)
        if os.path.exists(input_file):
            existing_files.append(filename)
    
    if not existing_files:
        pytest.skip("No test files found")
    
    # Effectuer la conversion en lot
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        input_path,
        existing_files,
        output_path
    )
    
    assert success, f"Batch conversion failed. Messages: {messages}"
    assert len(converted_files) == len(existing_files), f"Expected {len(existing_files)} converted files, got {len(converted_files)}"
    
    # Vérifier que tous les fichiers de sortie existent
    for converted_file in converted_files:
        assert os.path.exists(converted_file), f"Converted file not found: {converted_file}"
        assert converted_file.endswith('.h5'), f"Converted file should be .h5: {converted_file}"

def test_invalid_input_path(converter):
    """Test avec un chemin d'entrée invalide."""
    invalid_path = "invalid/path/that/does/not/exist"
    output_path = OUTPUT_DIR
    
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        invalid_path,
        FILES,
        output_path
    )
    
    assert not success, "Conversion should fail with invalid input path"
    assert len(converted_files) == 0, "No files should be converted with invalid input path"

def test_empty_file_list(converter):
    """Test avec une liste de fichiers vide."""
    input_path = TEST_DATA_DIR
    output_path = OUTPUT_DIR
    
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        input_path,
        [],
        output_path
    )
    
    assert not success, "Conversion should fail with empty file list"
    assert len(converted_files) == 0, "No files should be converted with empty file list"

def test_nonexistent_files(converter):
    """Test avec des fichiers qui n'existent pas."""
    input_path = TEST_DATA_DIR
    output_path = OUTPUT_DIR
    nonexistent_files = ["nonexistent1.cdf", "nonexistent2.cdf"]
    
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        input_path,
        nonexistent_files,
        output_path
    )
    
    assert not success, "Conversion should fail with nonexistent files"
    assert len(converted_files) == 0, "No files should be converted with nonexistent files"

# Tests pour les méthodes utilitaires
def test_get_files_from_folder(converter):
    """Test de la méthode get_files_from_folder."""
    if os.path.exists(TEST_DATA_DIR):
        files = converter.get_files_from_folder(TEST_DATA_DIR)
        cdf_files = [f for f in files if f.endswith('.cdf')]
        assert isinstance(files, list), "Should return a list"
        # Vérifier que seuls les fichiers .cdf sont retournés
        for f in files:
            assert f.endswith('.cdf'), f"Non-CDF file returned: {f}"
    else:
        pytest.skip(f"Test data directory not found: {TEST_DATA_DIR}")

def test_get_free_space(converter):
    """Test de la méthode get_free_space."""
    free_space = converter.get_free_space(".")
    assert isinstance(free_space, (int, float)), "Should return numeric value"
    assert free_space > 0, "Free space should be positive"

# Test d'intégration pour vérifier la structure HDF5
@pytest.mark.parametrize("filename", FILES)
def test_hdf5_structure(converter, filename):
    """Test pour vérifier la structure du fichier HDF5 généré."""
    input_path = TEST_DATA_DIR
    output_path = OUTPUT_DIR
    
    input_file = os.path.join(input_path, filename)
    if not os.path.exists(input_file):
        pytest.skip(f"Input file not found: {input_file}")
    
    success, messages, converted_files = converter.convert_cdf_to_hdf5_threaded(
        input_path,
        [filename],
        output_path
    )
    
    assert success, f"Conversion failed: {messages}"
    
    output_file = converted_files[0]
    
    # Vérifier la structure HDF5
    with h5py.File(output_file, "r") as h5f:
        expected_datasets = [
            'scan_acquisition_time',
            'mass_values',
            'intensity_values',
            'total_intensity',
            'point_count',
            'mass_range_min',
            'mass_range_max'
        ]
        
        # Vérifier que les datasets attendus sont présents
        for dataset_name in expected_datasets:
            if dataset_name in h5f:
                print(f"✅ Dataset found: {dataset_name}")
                # Vérifier que les données ne sont pas vides
                data = h5f[dataset_name][:]
                assert data.size > 0, f"Dataset {dataset_name} is empty"
            else:
                print(f"⚠️  Dataset not found: {dataset_name}")
        
        # Vérifier les attributs
        if 'scan_number_size' in h5f.attrs:
            scan_size = h5f.attrs['scan_number_size']
            assert isinstance(scan_size, (int, np.integer)), "scan_number_size should be integer"
            assert scan_size > 0, "scan_number_size should be positive"
            print(f"✅ Scan number size: {scan_size}")

if __name__ == "__main__":
    # Pour exécuter les tests directement
    pytest.main([__file__, "-v"])


    #run
    # cd interface_flask#
    #  pytest .\tests\test_converter.py -v
