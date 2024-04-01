from ner.configuration.gcloud import GCloud

obj = GCloud()

obj.sync_folder_from_gcloud(gcp_bucket_url ="ner-using-bert-24", filename="archive.zip", destination="test")