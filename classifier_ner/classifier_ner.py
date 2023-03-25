from transformers import AutoModelForTokenClassification
import torch

class Classifier:
    """
    This class performs the Named Entity Recognition
    """
    def __init__(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    def predict(self, df, inputs_column='inputs'):
        outputs = df[inputs_column].apply(lambda x:self.model(**x).logits)
        df['predictions'] = outputs.apply(lambda x:torch.argmax(x, dim=2))

        return df