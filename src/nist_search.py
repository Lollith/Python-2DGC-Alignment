import logging
import requests
import time
import pyms
from pyms_nist_search.search_result import SearchResult
import sys
import os
import requests
# from src.nist_utils.reference_data import ReferenceData
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nist_utils.reference_data import ReferenceData
import platform
import os


class NISTSearchWrapper:

    def __init__(self):
        logging.info("Initialisation du moteur NIST...")
        self.username = os.getenv("USERNAME")
        self.password = os.getenv("FLASK_PASSWORD")
        
        # if mode == "local"
        #     if platform.system() == "Windows":
        #         self.url = "http://host.docker.internal:8080/"
        #     else:
        #         self.url = "http://localhost:8080/"
        # else:
        self.url = "http://10.172.16.115:8080/"
        
        print(self.url)

    def check_nist_health(self):
        endpoint = f'{self.url}nist/health'
        try:
            #TODO
            #res = requests.get(endpoint, timeout=2, auth=(self.username, self.password))
            res = requests.get(endpoint, timeout=2)
            res.raise_for_status()
            health = res.json()
            return health['nist_status'] == 'available'
        except Exception as e:
            print(f"Erreur de connexion à NIST: {e}")
            return False
        
    def hit_list_from_nist_api(self, result_json):
        """
        Transforme le résultat JSON de l'API Flask NIST en liste de tuples
        (SearchResult, ReferenceData)
        """
        # for hit in result_json:
        #     print("type(hit) =", type(hit), "value =", hit)
        hit_list = []

        for hit in result_json:
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


    def nist_batch_search(self, list_serialized_spectrum):
        """
        Envoie une liste de spectres à l'API Flask NIST pour identification en lot.
        """
        # print(list_mass_spectra)
        endpoint = f'{self.url}nist/batch_search'
        print("ENDPOINT:", endpoint)
    
        # for spec in list_serialized_spectrum:
        #     print(f"Type reçu : {type(spec)}")
        # spectra_data = [spec.to_dict() for spec in list_mass_spectra]
        retry_count = 0
        while retry_count < 10:
            try:
                time.sleep(0.1)
                res = requests.post(
                    endpoint, json=list_serialized_spectrum,
                    auth=(self.username, self.password)
                    )
                res.raise_for_status()
                return res.json()
            
            except requests.exceptions.ConnectionError as e:
                print(f"Erreur de connexion (tentative {retry_count + 1}/10): {e}")
                time.sleep(0.5)
                retry_count += 1
            except requests.exceptions.Timeout as e:
                print(f"Timeout (tentative {retry_count + 1}/10): {e}")
                time.sleep(0.5)
                retry_count += 1
            except requests.exceptions.HTTPError as e:
                print(f"Erreur HTTP: {e}")
                print(f"Réponse: {res.text}")
                raise
        raise TimeoutError("Unable to communicate with the NIST API server.")


    
    def nist_single_search(self, serialized_spectrum):
        """
        Envoie un seul spectre à l'API Flask NIST pour identification.
        """
        endpoint = f'{self.url}nist/search'
        print(f"Requête vers {endpoint}")
        
        retry_count = 0
        while retry_count < 10:
            try:
                res = requests.post(
                    endpoint, json=serialized_spectrum,
                    auth=(self.username, self.password)
                )
                res.raise_for_status()
                # print("Réponse JSON brute :", res.json())
                return res.json()["hits"]

            except requests.exceptions.ConnectionError as e:
                print(f"Erreur de connexion (tentative {retry_count + 1}/10): {e}")
                time.sleep(0.5)
                retry_count += 1
            except requests.exceptions.Timeout as e:
                print(f"Timeout (tentative {retry_count + 1}/10): {e}")
                time.sleep(0.5)
                retry_count += 1
            except requests.exceptions.HTTPError as e:
                print(f"Erreur HTTP: {e}")
                print(f"Réponse: {res.text}")
                raise

        raise TimeoutError("Unable to communicate with the NIST API server.")

if __name__ == '__main__':

    response = requests.get("http://localhost:8080/")
    print(response.status_code)
    #print(response.text)

    response = requests.get("http://localhost:8080/routes")
    print(response.status_code)
    print(response.text)

    response = requests.get("http://localhost:8080/nist/health")
    print(response.status_code)
    print(response.text)
    nist_api = NISTSearchWrapper()
    if not nist_api.check_nist_health():
        print("NIST API is not available. Skipping search.")

