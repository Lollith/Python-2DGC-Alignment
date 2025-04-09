import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src import read_chroma
# import read_chroma  # Assurez-vous d'importer le bon fichier
import mass_spec
import skimage
import baseline_correction

# lancer : test/pytest test_read_chromato.py

@patch('read_chroma.read_chroma')  # Mock de la fonction read_chroma
@patch('mass_spec.read_full_spectra_centroid')  # Mock de read_full_spectra_centroid
@patch('read_chroma.full_spectra_to_chromato_cube')  # Mock de full_spectra_to_chromato_cube
@patch('baseline_correction.chromato_no_baseline')  # Mock de chromato_no_baseline
@patch('baseline_correction.chromato_cube_corrected_baseline')  # Mock de chromato_cube_corrected_baseline
@patch('skimage.restoration.estimate_sigma')  # Mock de estimate_sigma
def test_read_chromato_and_chromato_cube(
    mock_estimate_sigma,
    mock_chromato_cube_corrected_baseline,
    mock_chromato_no_baseline,
    mock_full_spectra_to_chromato_cube,
    mock_read_full_spectra_centroid,
    mock_read_chroma
):
    # Données mockées
    spectra_obj = (10, 20, np.array([1, 2]), np.array([1, 2]), 0, 100)
    mock_read_chroma.return_value = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), spectra_obj)
    mock_read_full_spectra_centroid.return_value = (np.array([1, 2, 3]), np.array([1, 2]), np.array([0, 1]))
    mock_full_spectra_to_chromato_cube.return_value = np.array([[1, 2], [3, 4]])
    mock_chromato_no_baseline.return_value = np.array([1, 2, 3])
    mock_chromato_cube_corrected_baseline.return_value = np.array([[1, 2], [3, 4]])
    mock_estimate_sigma.return_value = 0.5

    # Appel de la fonction à tester
    filename = 'test_file.xlsx'
    mod_time = 1.25
    pre_process = True

    chromato, time_rn, chromato_cube, sigma, (range_min, range_max) = (
        read_chroma.read_chromato_and_chromato_cube(filename, mod_time, pre_process))

    # Assertions classiques
    assert chromato is not None
    assert time_rn is not None
    assert chromato_cube is not None
    assert sigma == 0.5
    assert (range_min, range_max) == (0, 100)

    # Vérification des appels
    mock_read_chroma.assert_called_with(filename, mod_time)

    # Vérification des arguments passés à read_full_spectra_centroid
    _, kwargs = mock_read_full_spectra_centroid.call_args
    passed_spectra_obj = kwargs['spectra_obj']
    assert passed_spectra_obj[0] == spectra_obj[0]
    assert passed_spectra_obj[1] == spectra_obj[1]
    np.testing.assert_array_equal(passed_spectra_obj[2], spectra_obj[2])
    np.testing.assert_array_equal(passed_spectra_obj[3], spectra_obj[3])
    assert passed_spectra_obj[4] == spectra_obj[4]
    assert passed_spectra_obj[5] == spectra_obj[5]

    # Vérification de l'appel à estimate_sigma avec des tableaux
    args, kwargs = mock_estimate_sigma.call_args
    np.testing.assert_array_equal(args[0], np.array([1, 2, 3]))
    assert kwargs.get("channel_axis") is None


from unittest.mock import patch, MagicMock
import numpy as np
import read_chroma
import mass_spec
import skimage
import baseline_correction

# Fixtures pour le mock
@pytest.fixture
def mock_estimate_sigma():
    with patch('skimage.restoration.estimate_sigma') as mock:
        yield mock

@pytest.fixture
def mock_chromato_cube_corrected_baseline():
    with patch('baseline_correction.chromato_cube_corrected_baseline') as mock:
        yield mock

@pytest.fixture
def mock_chromato_no_baseline():
    with patch('baseline_correction.chromato_no_baseline') as mock:
        yield mock

@pytest.fixture
def mock_full_spectra_to_chromato_cube():
    with patch('read_chroma.full_spectra_to_chromato_cube') as mock:
        yield mock

@pytest.fixture
def mock_read_full_spectra_centroid():
    with patch('mass_spec.read_full_spectra_centroid') as mock:
        yield mock

@pytest.fixture
def mock_read_chroma():
    with patch('read_chroma.read_chroma') as mock:
        yield mock

# Test avec pre_process=False
def test_read_chromato_without_preprocessing(
    mock_estimate_sigma,
    mock_chromato_cube_corrected_baseline,
    mock_chromato_no_baseline,
    mock_full_spectra_to_chromato_cube,
    mock_read_full_spectra_centroid,
    mock_read_chroma
):
    spectra_obj = (10, 20, np.array([1, 2]), np.array([1, 2]), 0, 100)
    mock_read_chroma.return_value = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), spectra_obj)
    mock_read_full_spectra_centroid.return_value = (np.array([1, 2, 3]), np.array([1, 2]), np.array([0, 1]))
    mock_full_spectra_to_chromato_cube.return_value = np.array([[1, 2], [3, 4]])
    mock_estimate_sigma.return_value = 0.5

    filename = 'test_file.xlsx'
    mod_time = 1.25
    pre_process = False

    chromato, time_rn, chromato_cube, sigma, (range_min, range_max) = read_chroma.read_chromato_and_chromato_cube(
        filename, mod_time, pre_process
    )

    assert chromato is not None
    assert time_rn is not None
    assert chromato_cube is not None
    assert sigma == 0.5
    assert (range_min, range_max) == (0, 100)

    # Vérifie que les fonctions de baseline n'ont PAS été appelées
    mock_chromato_no_baseline.assert_not_called()
    mock_chromato_cube_corrected_baseline.assert_not_called()

# Test d'erreur fichier introuvable
@patch('read_chroma.read_chroma')
def test_read_chromato_file_not_found(mock_read_chroma):
    # Simuler une exception de fichier manquant
    mock_read_chroma.side_effect = FileNotFoundError("Fichier non trouvé")

    filename = 'invalid_file.xlsx'
    mod_time = 1.25
    pre_process = True

    with pytest.raises(FileNotFoundError, match="Fichier non trouvé"):
        read_chroma.read_chromato_and_chromato_cube(filename, mod_time, pre_process)
