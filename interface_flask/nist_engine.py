import logging
import pyms
import pyms_nist_search
import numpy as np
import os
import pyms.Spectrum

class NistEngine:
    """
    NIST Engine class to handle NIST-related operations.
    """

    def __init__(self):
        """
        Initialize the NIST Engine with provided NIST data.
        """
        logging.info("Initialisation du moteur NIST...")
        mainlib_path = os.getenv("MAINLIB_PATH", "C:/NIST20/MSSEARCH/mainlib")
        temp_dir = os.getenv("TEMP_DIR", "C:/NIST20/MSSEARCH/tmp")
        self.engine = pyms_nist_search.Engine(
            mainlib_path,
            pyms_nist_search.NISTMS_MAIN_LIB,
            temp_dir
        )

    def serialize_hit_tuple(self, hit_tuple):
        search_result, ref_data = hit_tuple
        return {
            "name": getattr(search_result, "name", None),
            "match_factor": getattr(search_result, "match_factor", None),
            "reverse_match": getattr(search_result, "reverse_match_factor", None),
            "hit_prob": getattr(search_result, "hit_prob", None),
            "cas_number": getattr(search_result, "cas", None),
            "spec_loc": getattr(search_result, "spec_loc", None),
            "formula": getattr(ref_data, "formula", None),
        }

    
    
    def full_search_with_ref_data(self, data):
        """
        Perform a search operation using the provided query.

        :param query: The search query to be executed.
        :return: Search results based on the query.
        """
        # Placeholder for search logic
        # This should interact with the NIST data and return results
        # mass_spectrum = pyms.Spectrum.MassSpectrum(mass_values, int_values)
        max_hits = 20
        try:
            mass = data["mass"]
            intensity = data["intensity"]
            spectrum = pyms.Spectrum.MassSpectrum(mass, intensity)
            hits = self.engine.full_search_with_ref_data(
                spectrum, max_hits)
            return [self.serialize_hit_tuple(hit) for hit in hits]
        except Exception as e:
            logging.error(f"Erreur lors de la conversion du spectre: {e}")
            raise
