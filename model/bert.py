import sys
import torch
from ner.exception import NerException
from transformers import BertForTokenClassification


class BertModel(torch.nn.Module):
    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            "bert-base-cased", num_labels=len(unique_labels)
        )

    def forward(self, input_id, mask, label):
        try:
            output = self.bert(
                input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
            )

            return output

        except Exception as e:
            raise NerException(e, sys) from e
