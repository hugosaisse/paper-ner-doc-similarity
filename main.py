from text_preprocessor.text_preprocessor import TextPreprocessorChunks
from classifier_ner.classifier_ner import Classifier
from sim_matrix_calculator.sim_matrix_calculator import SimilarityMatrixCalculator
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

    # Preprocess and vectorize the summarized text versions
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

# Calculate similarity matrices from the vectorized dataset
# Each column specified below represents a different text representation
# 1) full_text_emb_hs: full text vectorized using the last hidden states from the pre-trained model
# 2) summarized_text_O_emb_hs: summarized text removing predictions in ['O'] vectorized using the last hidden states from the pre-trained model
# 3) summarized_text_O_TEMPO_LOCAL_emb_hs: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL'] vectorized using the last hidden states from the pre-trained model
# 4) summarized_text_LEG_JURIS_emb_hs: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL', 'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO'] vectorized using the last hidden states from the pre-trained model
columns = ["full_text_emb_hs", "summarized_text_O_emb_hs", "summarized_text_O_TEMPO_LOCAL_emb_hs", "summarized_text_LEG_JURIS_emb_hs"]

similarity_matrix_obj = SimilarityMatrixCalculator(vectorized_df, columns)
similarity_matrix_obj.calculate_similarity_matrices()
similarity_matrix_obj.save_similarity_matrices()



