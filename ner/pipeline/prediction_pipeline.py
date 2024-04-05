import os
import sys
import torch
from ner.configuration.gcloud import GCloud
from ner.constants import *
from ner.entity.config_entity import ModelPredictorConfig
from ner.exception import NerException
from ner.logger import logging
from ner.utils.utils import MainUtils


class ModelPredictor:
    def __init__(self) -> None:
        self.model_predictor_config = ModelPredictorConfig()
        self.utils = MainUtils()
        self.gcloud = GCloud()


    def align_word_ids(self, texts: str, tokenizer: dict) -> list:
        logging.info("Entered the align_word_ids method of Model predictor class")
        try:
            label_all_tokens = False

            tokenized_inputs = tokenizer(
                texts, padding="max_length", max_length=512, truncation=True
            )

            word_ids = tokenized_inputs.word_ids()

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(1)
                    except:
                        label_ids.append(-100)
                else:
                    try:
                        label_ids.append(1 if label_all_tokens else -100)
                    except:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            logging.info("Exited the align_word_ids method of Model predictor class")
            return label_ids

        except Exception as e:
            raise NerException(e, sys) from e


    def evaluate_one_text(self, model: object, sentence: str, tokenizer: dict, ids_to_labels: dict) -> str:
        logging.info("Entered the evaluate_one_text method of Model predictor class")
        try:

            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            if use_cuda:
                model = model.cuda()

            text = tokenizer(
                sentence,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )

            mask = text["attention_mask"].to(device)
            input_id = text["input_ids"].to(device)
            label_ids = (
                torch.Tensor(self.align_word_ids(sentence, tokenizer))
                .unsqueeze(0)
                .to(device)
            )

            logits = model(input_id, mask, None)
            logits_clean = logits[0][label_ids != -100]

            predictions = logits_clean.argmax(dim=1).tolist()
            prediction_label = [ids_to_labels[i] for i in predictions]

            logging.info("Exited the evaluate_one_text method of Model predictor class")
            return sentence, prediction_label

        except Exception as e:
            raise NerException(e, sys) from e


    def initiate_model_predictor(self, sentence: str) -> str:
        logging.info(
            "Started Model Prediction >>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        try:
            os.makedirs(self.model_predictor_config.best_model_dir, exist_ok=True)
            logging.info(
                f"Created {os.path.basename(self.model_predictor_config.best_model_dir)} directory."
            )

            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=BUCKET_NAME,
                filename=TOKENIZER_FILE_NAME,
                destination=self.model_predictor_config.tokenizer_local_path,
            )
            logging.info("Tokenizer Pickle file downloaded from google storage")

            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=BUCKET_NAME,
                filename=IDS_TO_LABELS_FILE_NAME,
                destination=self.model_predictor_config.ids_to_labels_local_path,
            )
            logging.info("ids to label Pickle file downloaded from google storage")

            self.gcloud.sync_folder_from_gcloud(
                gcp_bucket_url=BUCKET_NAME,
                filename=GCP_MODEL_NAME,
                destination=self.model_predictor_config.best_model_from_gcp_path,
            )
            logging.info("Downloaded best model to Best_model directory.")

            tokenizer = self.utils.load_pickle_file(
                filepath=self.model_predictor_config.tokenizer_local_path
            )
            logging.info("Loaded tokenizer object")

            ids_to_labels = self.utils.load_pickle_file(
                filepath=self.model_predictor_config.ids_to_labels_local_path
            )
            logging.info("Loaded ids to lables file.")

            model = torch.load(self.model_predictor_config.best_model_path, map_location=torch.device('cpu'))
            logging.info("Best model loaded for prediction.")

            sentence, prediction_lable = self.evaluate_one_text(
                model=model,
                sentence=sentence,
                tokenizer=tokenizer,
                ids_to_labels=ids_to_labels,
            )

            return sentence, prediction_lable

        except Exception as e:
            raise NerException(e, sys) from e
