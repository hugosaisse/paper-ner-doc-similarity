from transformers import AutoModelForTokenClassification, pipeline
import pandas as pd
import torch

class Classifier:
    """
    This class performs the Named Entity Recognition
    """
    def __init__(self, model_path, device=torch.device("cpu")):
        self.model_path = model_path
        self.device = device
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
    
    def predict(self, df, inputs_column='inputs'):
        outputs_list = []
        for inputs in df[inputs_column]:
            outputs = self.model(**inputs.to(self.device))
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=2).tolist()[0]
            predicted_label = torch.argmax(logits, dim=2)
            p_list = []
            for p in predicted_label[0].cpu().numpy():
                p_list.append(self.model.config.id2label[p])
            outputs_list.append({'logits': logits.tolist()[0],
                                 'probabilities': probabilities,
                                 'predicted_label': predicted_label.tolist()[0],
                                 'prediction': p_list
                                })
        predicted_labels = [outputs['predicted_label'] for outputs in outputs_list]
        predictions = [outputs['prediction'] for outputs in outputs_list]
        df['labels'] = pd.Series(predicted_labels)
        df['predictions'] = pd.Series(predictions)

        return df