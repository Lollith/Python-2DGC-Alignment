import logger
import requests
import time
import pyms
from pyms_nist_search.search_result import SearchResult
from nist_utils.reference_data import ReferenceData

class NISTSearchWrapper:

    def __init__(self):
        logger.info("Initialisation du moteur NIST...")
        self.url = 'http://localhost:8080/'

    def check_nist_health(self):
        try:
            res = requests.get(self.url, timeout=2)
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
        endpoint = f'{self.url}/nist/batch_search'
        spectra_data = [spec.to_dict() for spec in list_mass_spectra] 
        retry_count = 0
        while retry_count < 10:
            try:
                res = requests.post(endpoint, json={"spectra": spectra_data})
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
        # Adapte selon tes classes SearchResult et ReferenceData
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
            # molecular_weight=hit.get("molecular_weight")
        )
        hit_list.append((search_result, ref_data))
    return hit_list
