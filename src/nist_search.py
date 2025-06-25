import logging
import requests
import time
import pyms
from pyms_nist_search.search_result import SearchResult
# from src.nist_utils.reference_data import ReferenceData
from src.nist_utils.reference_data import ReferenceData
import platform
from requests.auth import HTTPBasicAuth
import os
# import nist_search

# USERNAME = os.getenv("FLASK_USERNAME")
# PASSWORD = os.getenv("FLASK_PASSWORD")
# AUTH = HTTPBasicAuth(USERNAME, PASSWORD)
# auth = HTTPBasicAuth()
# hashed_password = os.getenv('FLASK_HASHED_PASSWORD')
# username = os.getenv('USERNAME')


class NISTSearchWrapper:

    def __init__(self):
        logging.info("Initialisation du moteur NIST...")
        self.username = os.getenv("USERNAME")
        self.password = os.getenv("FLASK_PASSWORD")
        
        if platform.system() == "Windows":
            self.url = "http://host.docker.internal:8080/"
        else:
            # Linux : à condition d’utiliser `network_mode: host`
            self.url = "http://localhost:8080/"

    def check_nist_health(self):
        endpoint = f'{self.url}nist/health'
        try:
            res = requests.get(endpoint, timeout=2, auth=(self.username, self.password))
            res.raise_for_status()
            health = res.json()
            return health['nist_status'] == 'available'
        except Exception as e:
            print(f"Erreur de connexion à NIST: {e}")
            return False

    def nist_batch_search(self, list_mass_spectra):
        """
        Envoie une liste de spectres à l'API Flask NIST pour identification en lot.
        """
        # print(list_mass_spectra)
        endpoint = f'{self.url}nist/batch_search'
        print(endpoint)

        def spectrum_to_dict(spectrum):
            return {
                "mass": [float(m) for m in spectrum.mass_list],
                "intensity": [float(i) for i in spectrum.intensity_list]
            }
        # spectra_data = [spec.to_dict() for spec in list_mass_spectra]
        spectra_data = [spectrum_to_dict(spec) for spec in list_mass_spectra]
        retry_count = 0
        while retry_count < 10:
            try:
                res = requests.post(endpoint, json={"spectra": spectra_data}, auth=(self.username, self.password))
                res.raise_for_status()
                return res.json()["results"]
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
                retry_count += 1
        raise TimeoutError("Unable to communicate with the NIST API server.")

    def hit_list_from_nist_api(self, result_json):
        """
        Transforme le résultat JSON de l'API Flask NIST en liste de tuples (SearchResult, ReferenceData)
        """
        hit_list = []
        for hit in result_json.get("hits", []):
            search_result = SearchResult(
                name=hit.get("name"),
                match_factor=hit.get("match_factor"),
                reverse_match_factor=hit.get("reverse_match"),
                hit_prob=hit.get("hit_prob"),
                cas=hit.get("cas_number"),
                spec_loc=hit.get("spec_loc")
            )
            ref_data = ReferenceData(
                formula=hit.get("formula"),
            )
            hit_list.append((search_result, ref_data))
        return hit_list


if __name__ == '__main__':
    nist_api = NISTSearchWrapper()
    if not nist_api.check_nist_health():
        print("NIST API is not available. Skipping search.")

