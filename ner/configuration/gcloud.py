import os
from zipfile import Path
import time


class GCloud:
    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):

        command = f"gsutil cp {filepath}/{filename} gs://{gcp_bucket_url}/"
        os.system(command)



    def sync_folder_from_gcloud(
        self, gcp_bucket_url: str, filename: str, destination: Path
    ):

        command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}"
        os.system(command)



