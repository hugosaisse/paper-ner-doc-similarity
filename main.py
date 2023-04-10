from text_preprocessor.text_preprocessor import TextPreprocessorChunks
from classifier_ner.classifier_ner import Classifier
from vectorizer.vectorizer import VectorizerHiddenStates, VectorizerInferenceAPI
import numpy as np
import pandas as pd
import torch

# Check that MPS (MacOS M1 Pro) is available, use GPU if available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
else:
    device = torch.device("mps")

# Preprocess and Tokenize
model_path = 'dominguesm/legal-bert-ner-base-cased-ptbr'
preprocessor = TextPreprocessorChunks(model_path)
df = pd.read_csv('./data/data.csv')
preprocessed_df = preprocessor.preprocess_text(df, raw_text_column='docContSplitted')

# Save preprocessed dataframe to CSV
preprocessed_df.to_csv('./data/preprocessed_data.csv', index=False)

# Get predictions (Named Entities)
classifier = Classifier(model_path, device)
classified_df = classifier.predict(preprocessed_df)

# Save classified dataframe to CSV
classified_df.to_csv('./data/classified_data.csv', index=False)

#henrique.asp.mentzingen
#souza.www.com.br

# Vectorize the preprocessed text considering the whole sentences (without extracting legislation and jurisprudence)
# Using feature extraction (last hidden states from pre-trained model)
text_vectorizer = VectorizerHiddenStates(model_path, embedding_col='full_text_emb_hs', inputs_col='inputs', device=device)
vectorized_dataset = text_vectorizer.process_dataset(classified_df)
# Vectorize the preprocessed text using the Inference API
api_key = "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"
text_vectorizer_api = VectorizerInferenceAPI(model_path, embedding_col='full_text_emb_inf', tokens_col='tokens', api_key=api_key)
vectorized_dataset_2 = text_vectorizer_api.process_dataset(classified_df)

# Flatten list of lists containing tokens, labels and predictions
flattened_df = classified_df.copy(deep=True)

flattened_df['tokens'] = flattened_df['tokens'].apply(lambda x: [item for sublist in x for item in sublist])
flattened_df['labels'] = flattened_df['labels'].apply(lambda x: [item for sublist in x for item in sublist])
flattened_df['predictions'] = flattened_df['predictions'].apply(lambda x: [item for sublist in x for item in sublist])

# List infractions evaluated by experts (for holdout sample)
df_expert1 = pd.read_excel('./data/similarity_experts_evaluation_i.xlsm', sheet_name='Similaridade')
holdout_violations = np.unique(df_expert1[['Inf A', 'Inf B']].values)

# Select infractions for fine-tuning the Word2Vec FastText model
df_train = preprocessed_df[~preprocessed_df['infracaoId'].isin(holdout_violations)]
df_train.reset_index(drop=True, inplace=True)


