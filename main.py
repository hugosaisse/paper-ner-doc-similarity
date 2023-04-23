from text_preprocessor.text_preprocessor import TextPreprocessorChunks
from classifier_ner.classifier_ner import Classifier
from summarizer.summarizer import Summarizer
from vectorizer.vectorizer import VectorizerHiddenStates, VectorizerInferenceAPI
import os
import pandas as pd
import pickle
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

model_path = 'dominguesm/legal-bert-ner-base-cased-ptbr'

# Preprocess and tokenize the text if it hasn't been done yet
if not os.path.isfile('./data/preprocessed_data.pkl'):
    # Preprocess and Tokenize
    preprocessor = TextPreprocessorChunks(model_path)
    df = pd.read_csv('./data/data.csv')
    preprocessed_df = preprocessor.preprocess_text(df, raw_text_column='docContSplitted')
    # Save preprocessed dataframe to pickle
    with open('./data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_df, f)
else:
    # Load preprocessed dataframe from pickle
    with open('./data/preprocessed_data.pkl', 'rb') as f:
        preprocessed_df = pickle.load(f)

# Classify (return named entities) the text if it hasn't been done yet
if not os.path.isfile('./data/classified_data.pkl'):
    # Get predictions (Named Entities)
    classifier = Classifier(model_path, device)
    classified_df = classifier.predict(preprocessed_df)
    # Save classified dataframe to pickle
    with open('./data/classified_data.pkl', 'wb') as f:
        pickle.dump(classified_df, f)
else:
    with open('./data/classified_data.pkl', 'rb') as f:
        classified_df = pickle.load(f)

# After classifying the text, we can summarize it by selecting the text
# that is not a named entity or by selecting the predictions we want to filter out
if not os.path.isfile('./data/summarized_data.pkl'):
    labels_to_filter = ['O']
    summarizer = Summarizer(model_path, labels_to_filter)
    summarized_df = summarizer.create_summarized_text(classified_df, output_col='summarized_text_O')
    
    labels_to_filter = ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL']
    summarizer = Summarizer(model_path, labels_to_filter)
    summarized_df = summarizer.create_summarized_text(summarized_df, output_col='summarized_text_O_TEMPO_LOCAL')
    
    labels_to_filter = ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL',
                        'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO']
    summarizer = Summarizer(model_path, labels_to_filter)
    summarized_df = summarizer.create_summarized_text(summarized_df, output_col='summarized_text_LEG_JURIS')

    # Save classified dataframe with summarized text to pickle
    with open('./data/summarized_data.pkl', 'wb') as f:
        pickle.dump(summarized_df, f)
else:
    with open('./data/summarized_data.pkl', 'rb') as f:
        summarized_df = pickle.load(f)

# Preprocess and vectorize the summarized text if it hasn't been done yet
# Vectorize the full text if it hasn't been done yet
if not os.path.isfile('./data/vectorized_dataset.pkl'):
    # Vectorize the preprocessed text considering the whole sentences (without extracting legislation and jurisprudence)
    # Using feature extraction (last hidden states from pre-trained model)
    text_vectorizer = VectorizerHiddenStates(model_path,
                                             embedding_col='full_text_emb_hs',
                                             inputs_col='inputs',
                                             device=device)
    vectorized_df = text_vectorizer.process_dataset(summarized_df)

    # Preprocess and vectorize the summarized text
    # Preprocess and Tokenize
    preprocessor = TextPreprocessorChunks(model_path)
    for column in summarized_df.columns:
        if 'summarized_text' in column:
            preprocessed_df = preprocessor.preprocess_text(summarized_df,
                                                           raw_text_column=column)
            text_vectorizer = VectorizerHiddenStates(model_path,
                                             embedding_col=column + '_emb_hs',
                                             inputs_col='inputs',
                                             device=device)
            v_df = text_vectorizer.process_dataset(preprocessed_df)
            vectorized_df = pd.concat([vectorized_df, v_df[column + '_emb_hs']], axis=1)
    # Save vectorized dataset to pickle
    with open('./data/vectorized_dataset.pkl', 'wb') as f:
        pickle.dump(vectorized_df, f)
else:
    # Load vectorized dataset from pickle
    with open('./data/vectorized_dataset.pkl', 'rb') as f:
        vectorized_df = pickle.load(f)
    
"""
# Vectorize the preprocessed text using the Inference API
api_key = "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"
text_vectorizer_api = VectorizerInferenceAPI(model_path, embedding_col='full_text_emb_inf', tokens_col='tokens', api_key=api_key)
vectorized_dataset_2 = text_vectorizer_api.process_dataset(classified_df)
# Save vectorized dataset 2 to CSV
vectorized_dataset_2.to_csv('./data/vectorized_dataset_2.csv', index=False)

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
"""


