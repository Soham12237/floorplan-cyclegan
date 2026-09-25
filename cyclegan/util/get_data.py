from __future__ import print_function
from pathlib import Path
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup


class GetData(object):
    

    def __init__(self, technique="cyclegan", verbose=True):
        url_dict = {
            "pix2pix": "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/",
            "cyclegan": "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets",
        }
        self.url = url_dict.get(technique.lower())
        self._verbose = verbose

    def _print(self, text):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, "lxml")
        options = [h.text for h in soup.find_all("a", href=True) if h.text.endswith((".zip", "tar.gz"))]
        return options

    def _present_options(self):
        r = requests.get(self.url)
        options = self._get_options(r)
        print("Options:\n")
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input("\nPlease enter the number of the " "dataset above you wish to download:")
        return options[int(choice)]

    def _download_data(self, dataset_url, save_path):
        save_path = Path(save_path)
        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

        base = Path(dataset_url).name
        temp_save_path = save_path / base

        with open(temp_save_path, "wb") as f:
            r = requests.get(dataset_url)
            f.write(r.content)

        if base.endswith(".tar.gz"):
            obj = tarfile.open(temp_save_path)
        elif base.endswith(".zip"):
            obj = ZipFile(temp_save_path, "r")
        else:
            raise ValueError("Unknown File Type: {0}.".format(base))

        self._print("Unpacking Data...")
        obj.extractall(save_path)
        obj.close()
        temp_save_path.unlink()

    def get(self, save_path, dataset=None):
        
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset

        save_path_full = Path(save_path) / selected_dataset.split(".")[0]

        if save_path_full.is_dir():
            warn(f"\n'{save_path_full}' already exists. Voiding Download.")
        else:
            self._print("Downloading Data...")
            url = f"{self.url}/{selected_dataset}"
            self._download_data(url, save_path=save_path)

        return save_path_full.resolve()
