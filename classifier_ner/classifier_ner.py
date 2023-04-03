from transformers import AutoModelForTokenClassification, BatchEncoding
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
            logits = []
            probabilities = []
            predicted_labels = []
            predictions = []
            for i in range(len(inputs['input_ids'])):
                ith_inputs = {k: v[i] for k, v in inputs.items()}
                ith_outputs = self.model(**BatchEncoding(ith_inputs).to(self.device))
                ith_logits = ith_outputs.logits
                ith_probabilities = torch.softmax(ith_logits, dim=2).tolist()[0]
                ith_predicted_labels = torch.argmax(ith_logits, dim=2)
                p_list = []
                
                for p in ith_predicted_labels[0].cpu().numpy():
                    p_list.append(self.model.config.id2label[p])
                
                logits.append(ith_logits.tolist()[0])
                probabilities.append(ith_probabilities)
                predicted_labels.append(ith_predicted_labels.tolist()[0])
                predictions.append(p_list)

            outputs_list.append({'logits': logits,
                                'probabilities': probabilities,
                                'predicted_labels': predicted_labels,
                                'predictions': predictions
                            })
        predicted_labels = [outputs['predicted_labels'] for outputs in outputs_list]
        predictions = [outputs['predictions'] for outputs in outputs_list]
        df['labels'] = pd.Series(predicted_labels)
        df['predictions'] = pd.Series(predictions)

        return df