# from unittest.mock import patch, Mock, MagicMock
from src import read_chroma
import pytest
import numpy as np
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
    mock_ds.__getitem__.side_effect = lambda key: {
        'total_intensity': np.array([
            [15974499.63954097, 15943862.03871114, 15939385.98558136, 15932633.86287273, 15931075.29439884],
            [14224064.43580875, 14258600.26621146, 14247219.02366124, 14220521.99589013, 14195360.93871526],
            [12119841.14841838, 12103128.67426359, 12075977.79191036, 12087192.97973525, 12055330.0499834],
            [11216531.68299363, 11137381.8642094,  11104735.58416802, 11120313.21237555, 11100312.31730843]
            ]),
        'scan_acquisition_time': np.array([510.0, 510.00832, 510.01664, 510.02496]),
        'mass_values': np.array([39.04689572, 39.89312009, 39.94080997, 40.05694416]),
        'intensity_values': np.array([28194.81086097,  318942.62786964, 541638.21430383, 43201.84535235]),
        'mass_range_min': np.array([39.01671406, 39.01671406, 39.01671406, 39.01671406]),
        'mass_range_max': np.array([500.99517556, 500.99517556, 500.99517556, 500.99517556]),
        'point_count': np.array([245, 227, 222, 218])
    }[key]

    # Appeler la fonction avec un fichier fictif
    result = read_chroma.read_chroma("test_file.cdf", 1.25)
    # Vérifier que le Dataset a été appelé correctement
    mock_dataset.assert_called_once_with("test_file.cdf")
    # Vérifier le format des données retournées par la fonction
    tic_chromato, (start_time, end_time), (l1, l2, mv, iv, range_min, range_max) = result

    # Vérifier les valeurs générées
    # assert tic_chromato[:4, :5].tolist() == [[15974499.63954097, 15943862.03871114, 15939385.98558136, 15932633.86287273, 15931075.29439884],
    #                                  [14224064.43580875, 14258600.26621146, 14247219.02366124, 14220521.99589013, 14195360.93871526],
    #                                  [12119841.14841838, 12103128.67426359, 12075977.79191036, 12087192.97973525, 12055330.0499834],
    #                                  [11216531.68299363, 11137381.8642094,  11104735.58416802, 11120313.21237555, 11100312.31730843]]


    assert start_time == 8.5
    assert end_time == 8.500416
    assert mv.tolist() == [39.04689572, 39.89312009, 39.94080997, 40.05694416]
    assert iv.tolist() == [28194.81086097, 318942.62786964, 541638.21430383, 43201.84535235]
    assert range_min == 40
    assert range_max == 500
    assert mock_ds["point_count"][:].tolist() == [245, 227, 222, 218]
