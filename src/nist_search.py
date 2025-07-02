import logging
import requests
import time
import pyms
from pyms_nist_search.search_result import SearchResult
import sys
import os
# from src.nist_utils.reference_data import ReferenceData
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nist_utils.reference_data import ReferenceData
import platform
import os
import requests


class NISTSearchWrapper:

    def __init__(self):
        logging.info("Initialisation du moteur NIST...")
        self.username = os.getenv("USERNAME")
        self.password = os.getenv("FLASK_PASSWORD")
        
        self.url = "http://10.172.16.115:8080/" #TODO
        # if platform.system() == "Windows":
        # self.url = "http://host.docker.internal:8080/" #TODO 
        # else:
        #     # Linux : à condition d’utiliser `network_mode: host`
        #     self.url = "http://localhost:8080/"

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

    # def nist_batch_search(self, list_mass_spectra):
    #     """
    #     Envoie une liste de spectres à l'API Flask NIST pour identification en lot.
    #     """
    #     # print(list_mass_spectra)
    #     endpoint = f'{self.url}nist/batch_search'
    #     print(endpoint)

    #     def spectrum_to_dict(spectrum):
    #         return {
    #             "mass": [float(m) for m in spectrum.mass_list],
    #             "intensity": [float(i) for i in spectrum.intensity_list]
    #         }
    #     for spec in list_mass_spectra:
    #         print(f"Type reçu : {type(spec)}")
    #     # spectra_data = [spec.to_dict() for spec in list_mass_spectra]
    #     spectra_data = [spectrum_to_dict(spec) for spec in list_mass_spectra]
    #     retry_count = 0
    #     while retry_count < 10:
    #         try:
    #             res = requests.post(
    #                 endpoint, json={"spectra": spectra_data},
    #                 auth=(self.username, self.password)
    #                 )
    #             res.raise_for_status()
    #             return res.json()["results"]
    #         except requests.exceptions.ConnectionError:
    #             time.sleep(0.5)
    #             retry_count += 1
    #     raise TimeoutError("Unable to communicate with the NIST API server.")

    def hit_list_from_nist_api(self, result_json):
        """
        Transforme le résultat JSON de l'API Flask NIST en liste de tuples
        (SearchResult, ReferenceData)
        """
        # for hit in result_json:
            # print("type(hit) =", type(hit), "value =", hit)  # <--- debug i
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
    
    def nist_single_search(self, serialized_spectrum):
        """
        Envoie un seul spectre à l'API Flask NIST pour identification.
        """
        endpoint = f'{self.url}nist/search'
        print(f"Requête vers {endpoint}")
        
        # def spectrum_to_dict(spectrum):
        #     return {
        #         "mass": [float(m) for m in spectrum.mass_list],
        #         "intensity": [float(i) for i in spectrum.intensity_list]
        #     }
        # data = [spec.to_dict() for spec in mass_spectrum]
        # data = spectrum_to_dict(spectrum=mass_spectrum)
        #print(f"données envoyées : {data}")
        # data = spectrum_to_dict(mass_spectrum)
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

