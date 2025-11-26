import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
class SimpleMultiTaskRoBERTa(nn.Module):
    def __init__(self, num_emotions, dropout=0.1):
        super(SimpleMultiTaskRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        
        # Simple direct classification heads
        self.sarcasm_classifier = nn.Linear(768, 2)
        self.emotion_classifier = nn.Linear(768, 8)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        sarcasm_logits = self.sarcasm_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)

        return sarcasm_logits, emotion_logits