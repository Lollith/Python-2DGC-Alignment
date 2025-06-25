import logging
import pyms
import pyms_nist_search
import numpy as np


class NistEngine:
    """
    NIST Engine class to handle NIST-related operations.
    """

    def __init__(self):
        """
        Initialize the NIST Engine with provided NIST data.

        :param nist_data: Data related to NIST standards and guidelines.
        
        """
        logging.info("Initialisation du moteur NIST...")
        mainlib_path = "C:/NIST20/MSSEARCH/mainlib/"
        temp_dir = "C:/NIST20/MSSEARCH/temp/"
        self.search = pyms_nist_search.Engine(
            mainlib_path,
            pyms_nist_search.NISTMS_MAIN_LIB,
            temp_dir
        )


    def spectrum_with_ref_data(self, mass_spectrum):
        """
        Perform a search operation using the provided query.

        :param query: The search query to be executed.
        :return: Search results based on the query.
        """
        # Placeholder for search logic
        # This should interact with the NIST data and return results
        # mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        hits = self.search.full_search_with_ref_data(mass_spectrum)
        
        print("hits", hits)
        return hits
    