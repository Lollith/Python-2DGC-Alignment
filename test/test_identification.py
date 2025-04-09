# from unittest.mock import patch, Mock, MagicMock
from src import read_chroma
import pytest
# pytest test/test_identification.py 

# read_chroma---------------------------------


def test_read_chroma_refuses_non_cdf(mocker):
    # remmplacer la fonction Dataset pour ne pas charger un vrai .cdf
    mock_dataset = mocker.patch("src.read_chroma.nc.Dataset")
    with pytest.raises(ValueError):
        read_chroma.read_chroma("test_file.txt", 1.25)
    mock_dataset.assert_not_called()


def test_read_chroma(mocker):
    mock_dataset = mocker.patch("src.read_chroma.nc.Dataset")
    # crée un objet mock qui simule un fichier .cdf
    mock_ds = mocker.MagicMock()

    mock_dataset.return_value = mock_ds

    # Simuler les valeurs retournées par le Dataset
    mock_ds.__getitem__.side_effect = {
        'total_intensity': [15974499.63954097, 15943862.03871114, 15939385.98558136, 15932633.86287273],
        'scan_acquisition_time': [510.0, 510.00832, 510.01664, 510.02496],
        'mass_values': [39.04689572, 39.89312009, 39.94080997, 40.05694416 ],
        'intensity_values': [28194.81086097,  318942.62786964, 541638.21430383, 43201.84535235],
        'mass_range_min': [39.01671406, 39.01671406, 39.01671406, 39.01671406],
        'mass_range_max': [500.99517556, 500.99517556, 500.99517556, 500.99517556],
        'point_count': "[245, 227, 222, 218]"
    }

    # Appeler la fonction avec un fichier fictif
    result = read_chroma.read_chroma("test_file.cdf", 1.25)

    # Vérifier que le Dataset a été appelé correctement
    mock_dataset.assert_called_once_with("test_file.cdf")

    # Vérifier le format des données retournées par la fonction
    tic_chromato, (start_time, end_time), (l1, l2, mv, iv, range_min, range_max) = result

    # Vérifier les valeurs générées
    assert tic_chromato == [[15974499.63954097, 15943862.03871114], [15939385.98558136, 15932633.86287273]]
    assert start_time == 510.0
    assert end_time == 510.02496
    assert mv == [39.04689572, 39.89312009, 39.94080997, 40.05694416]
    assert iv == [28194.81086097,  318942.62786964, 541638.21430383, 43201.84535235]
    assert range_min == 39.01671406
    assert range_max == 500.99517556
    assert mock_ds["point_count"][:].tolist() == [245, 227, 222, 218]
