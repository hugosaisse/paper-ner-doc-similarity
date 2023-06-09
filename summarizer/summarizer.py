import pandas as pd
from transformers import AutoTokenizer

class Summarizer:
    def __init__(self, model_path, labels_to_filter):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.labels_to_filter = labels_to_filter

    def create_summarized_text(self, ds, tokens_col='tokens', predictions_col='predictions', output_col='summarized_text'):
        summarized_texts = []

        df = pd.DataFrame() 
        
        # Flatten the list of lists containing tokens and predictions
        df[tokens_col] = ds[tokens_col].apply(lambda x: [item for sublist in x for item in sublist])
        df[predictions_col] = ds[predictions_col].apply(lambda x: [item for sublist in x for item in sublist])

        for index, row in df.iterrows():
            
            tokens = row[tokens_col]
            predictions = row[predictions_col]

            # Add special tokens to the list of labels to filter
            self.labels_to_filter.extend([self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.unk_token])

            # Filter out tokens with predictions in the labels_to_filter list
            filtered_tokens = [token for token, prediction in zip(tokens, predictions) if prediction not in self.labels_to_filter]

            # Convert the list of filtered tokens into a string using the tokenizer
            summarized_text = self.tokenizer.convert_tokens_to_string(filtered_tokens,
                                                                      #skip_special_tokens=True,
                                                                      #clean_up_tokenization_spaces=True
                                                                      )
            summarized_texts.append(summarized_text)

        # Add the summarized_texts as a new column to the DataFrame
        ds[output_col] = pd.Series(summarized_texts)

        return ds