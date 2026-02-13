from kaggle.api.kaggle_api_extended import KaggleApi
import os
from pathlib import Path

class Datafetcher():
    """
    Service object to manage authentication, downloading, and extraction of datasets.
    """
    def download_data(self, data_dir: Path) -> None:
        api = KaggleApi()
        api.authenticate()
        
        #setup path
        competition = "home-credit-default-risk"
        
        print(f"Downloading data for {competition}...")
        api.competition_download_files(competition, path=str(data_dir))
        
        import zipfile
        for file in os.listdir(data_dir):
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(data_dir, file), "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                os.remove(os.path.join(data_dir, file))
