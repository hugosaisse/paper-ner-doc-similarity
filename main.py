from text_preprocessor.text_preprocessor import TextPreprocessorChunks
from classifier_ner.classifier_ner import Classifier
from sim_matrix_calculator.sim_matrix_calculator import SimilarityMatrixCalculator
from summarizer.summarizer import Summarizer
from vectorizer.vectorizer import VectorizerHiddenStates, VectorizerInferenceAPI, VectorizerEmbeddingOpenAI
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

summ_models = {
    'dominguesm/legal-bert-ner-base-cased-ptbr': {
        'type': 'ner',
        'labels_to_filter': {
            'O':['O'],
            'O_TEMPO_LOCAL':['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL'],
            'LEG_JURIS':['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL', 'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO']
        },
        'labels_to_keep': {
            'O': ["B-ORGANIZACAO", "I-ORGANIZACAO", "B-PESSOA", "I-PESSOA", "B-TEMPO", "I-TEMPO", "B-LOCAL", "I-LOCAL", "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"],
            'O_TEMPO_LOCAL': ["B-ORGANIZACAO", "I-ORGANIZACAO", "B-PESSOA", "I-PESSOA", "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"],
            'LEG_JURIS': ["B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"]
        }
    },
    'lisaterumi/postagger-portuguese': {
        'type': 'pos',
        'labels_to_filter': None,
        'labels_to_keep': {
            'N': ['N'],
            'N_V': ['N', 'V']
        }
    }
}

openai_models = {
    'text-embedding-ada-002': {
        'type': 'embedding'
    }
}

# Create the "summ_model" folder under "models" if it does not exist
for summ_model in summ_models.keys():
    if not os.path.exists(f"models/{summ_model}"):
        os.makedirs(f"models/{summ_model}")
        os.makedirs(f"models/{summ_model}/data")
        os.makedirs(f"models/{summ_model}/sim_matrices")
    for embedding_model in openai_models.keys():
        if not os.path.exists(f"models/{summ_model}/{embedding_model}"):
            os.makedirs(f"models/{summ_model}/{embedding_model}")

# model_path is the base model choice
# It is used to tokenize the text, classify the tokens, summarize and vectorize the text
model_path = 'lisaterumi/postagger-portuguese'
# openai_model is the OpenAI's model chose to be used as an alternative to vectorize the text
openai_model = 'text-embedding-ada-002'

# Preprocess and Tokenize
# list of regex to remove expressions that don't add meaning to the text
# 0: 'coordenação (geral) de'
# 1: 'ministério da fazenda|economia'
# 2: 'superintendência de seguros privados'
# 3: brazilian postal code
# 4: URL
# 5: brazilian phone numbers
# 6: brazilian addresses
# 7: e-mail (emailregex.com)
regex_list = [r'(coordena[cç][aã]o)\s?-?(geral)?\s?de', r'(minist[eé]rio)\s(da)\s(fazenda|economia)',
r'(superintend[eê]ncia)\sde\sseguros\sprivados', r'(cep)\:?\s*\b\d{2}\.?\d{3}-?\d{3}\b',
r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
r'\b(\+55\s?)?\(?([0-9]{2,3}|0((x|[0-9]){2,3}[0-9]{2}))\)?\s*[0-9]{4,5}[- ]*[0-9]{4}\b',
r'\b(rua|r\.|avenida|av\.?|travessa|trav\.?|largo|quadra|qd|alameda|conjunto|conj\.?|estrada|pra[cç]a|rodovia|rod\.?)\s([a-zA-Z_\s]+)[, ]+(\d+)\s?([-/\da-zDA-Z\\ ]+)?\b',
r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
]

# Preprocess and tokenize the text if it hasn't been done yet
if not os.path.isfile(f'./models/{model_path}/data/preprocessed_data.pkl'):
    preprocessor = TextPreprocessorChunks(model_path, regex_list=regex_list)
    df = pd.read_csv('./data/data.csv')
    preprocessed_df = preprocessor.preprocess_text(df, raw_text_column='docContSplitted')
    # Save preprocessed dataframe to pickle
    with open(f'./models/{model_path}/data/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_df, f)
else:
    # Load preprocessed dataframe from pickle
    with open(f'./models/{model_path}/data/preprocessed_data.pkl', 'rb') as f:
        preprocessed_df = pickle.load(f)

# Classify (return named entities or token tags) the text if it hasn't been done yet
if not os.path.isfile(f'./models/{model_path}/data/classified_data.pkl'):
    # Get predictions (Named Entities)
    classifier = Classifier(model_path, device)
    classified_df = classifier.predict(preprocessed_df)
    # Save classified dataframe to pickle
    with open(f'./models/{model_path}/data/classified_data.pkl', 'wb') as f:
        pickle.dump(classified_df, f)
else:
    with open(f'./models/{model_path}/data/classified_data.pkl', 'rb') as f:
        classified_df = pickle.load(f)

# After classifying the text, we can summarize it by selecting the text
# that is not a named entity or by selecting the predictions we want to filter out
if not os.path.isfile(f'./models/{model_path}/data/summarized_data.pkl'):
    for tag, labels_to_keep in summ_models[model_path]['labels_to_keep'].items():
        labels_to_filter_series = classified_df['predictions'].apply(lambda x: [item for sublist in x for item in sublist if item not in labels_to_keep])
        labels_to_filter = pd.unique(labels_to_filter_series.explode()).tolist()
        summarizer = Summarizer(model_path, labels_to_filter=labels_to_filter)
        summarized_df = summarizer.create_summarized_text(classified_df, output_col=f'summarized_text_{tag}')
    
    # labels_to_filter = ['O']
    # summarizer = Summarizer(model_path, labels_to_filter)
    # summarized_df = summarizer.create_summarized_text(classified_df, output_col='summarized_text_O')
    
    # labels_to_filter = ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL']
    # summarizer = Summarizer(model_path, labels_to_filter)
    # summarized_df = summarizer.create_summarized_text(summarized_df, output_col='summarized_text_O_TEMPO_LOCAL')
    
    # labels_to_filter = ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL',
    #                     'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO']
    # summarizer = Summarizer(model_path, labels_to_filter)
    # summarized_df = summarizer.create_summarized_text(summarized_df, output_col='summarized_text_LEG_JURIS')

    # Save classified dataframe with summarized text to pickle
    with open(f'./models/{model_path}/data/summarized_data.pkl', 'wb') as f:
        pickle.dump(summarized_df, f)
else:
    with open(f'./models/{model_path}/data/summarized_data.pkl', 'rb') as f:
        summarized_df = pickle.load(f)

# Preprocess and vectorize the summarized text if it hasn't been done yet
# Vectorize the full text if it hasn't been done yet
if not os.path.isfile(f'./models/{model_path}/data/vectorized_hs_dataset.pkl'):
    # Vectorize the preprocessed text considering the whole sentences (without extracting legislation and jurisprudence)
    # Using feature extraction (last hidden states from pre-trained model)
    text_vectorizer = VectorizerHiddenStates(model_path,
                                             embedding_col='full_text_emb_hs',
                                             inputs_col='inputs',
                                             device=device)
    vectorized_hs_df = text_vectorizer.process_dataset(summarized_df)

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
            vectorized_hs_df = pd.concat([vectorized_hs_df, v_df[column + '_emb_hs']], axis=1)
    # drop duplicate columns (keep first by default)
    vectorized_hs_df = vectorized_hs_df.loc[:,~vectorized_hs_df.columns.duplicated()]
    # Save vectorized dataset to pickle
    with open(f'./models/{model_path}/data/vectorized_hs_dataset.pkl', 'wb') as f:
        pickle.dump(vectorized_hs_df, f)
else:
    # Load vectorized dataset from pickle
    with open(f'./models/{model_path}/data/vectorized_hs_dataset.pkl', 'rb') as f:
        vectorized_hs_df = pickle.load(f)

#Repeat the process using the inference API for vectorization
if not os.path.isfile(f'./models/{model_path}/data/vectorized_if_dataset.pkl'):
    # Vectorize the preprocessed text considering the whole sentences (without extracting legislation and jurisprudence)
    # Using feature extraction (inference API)
    with open("./data/hf_token.txt") as f:
        hf_token = f.read()
    with open(f"./models/{model_path}/api_url.txt") as f:
        api_url = f.read()

    preprocessor = TextPreprocessorChunks(model_path, regex_list=regex_list, max_length=100)
    preprocessed_df = preprocessor.preprocess_text(summarized_df,
                                                   raw_text_column='docContSplitted'
                                                  )

    text_vectorizer = VectorizerInferenceAPI(embedding_col='full_text_emb_if',
                                             sentences_col='sentences',
                                             hf_token=hf_token,
                                             api_url=api_url)
    vectorized_if_df = text_vectorizer.process_dataset(preprocessed_df)

    # Preprocess and vectorize the summarized text versions
    
    for column in summarized_df.columns:
        if 'summarized_text' in column:
            preprocessed_df = preprocessor.preprocess_text(summarized_df,
                                                           raw_text_column=column)
            text_vectorizer = VectorizerInferenceAPI(embedding_col=column + '_emb_if',
                                                     sentences_col='sentences',
                                                     hf_token=hf_token,
                                                     api_url=api_url)
            v_df = text_vectorizer.process_dataset(preprocessed_df)
            vectorized_if_df = pd.concat([vectorized_if_df, v_df[column + '_emb_if']], axis=1)
    # drop duplicate columns (keep first by default)
    vectorized_if_df = vectorized_if_df.loc[:,~vectorized_if_df.columns.duplicated()]
    # Save vectorized dataset to pickle
    with open(f'./models/{model_path}/data/vectorized_if_dataset.pkl', 'wb') as f:
        pickle.dump(vectorized_if_df, f)
else:
    # Load vectorized dataset from pickle
    with open(f'./models/{model_path}/data/vectorized_if_dataset.pkl', 'rb') as f:
        vectorized_if_df = pickle.load(f)

# Repeat the process using the embedding API from Open AI
tag = 'ada'
if not os.path.isfile(f'./models/{model_path}/{openai_model}/data/vectorized_{tag}_dataset.pkl'):
    # Vectorize the preprocessed text considering the whole sentences
    # (without extracting legislation and jurisprudence)
    with open("./data/openai_organization.txt") as f:
        openai_organization = f.read()
    with open("./data/openai_token.txt") as f:
        openai_token = f.read()

    preprocessor = TextPreprocessorChunks(model_path)
    preprocessed_df = summarized_df.copy(deep=True)
    preprocessed_df['sentences'] = preprocessed_df['docContSplitted'].apply(lambda x: preprocessor._regex_substitute(x, regex_list=regex_list))
    text_vectorizer = VectorizerEmbeddingOpenAI(embedding_col='full_text_emb_ada',
                                             sentences_col='sentences',
                                             openai_organization=openai_organization,
                                             openai_token=openai_token
                                            )
    vectorized_ada_df = text_vectorizer.process_dataset(preprocessed_df)

    # Preprocess and vectorize the summarized text versions
    for column in preprocessed_df.columns:
        if 'summarized_text' in column:
            preprocessed_df['sentences'] = preprocessed_df[column].apply(lambda x: preprocessor._regex_substitute(x, regex_list=regex_list))
            text_vectorizer = VectorizerEmbeddingOpenAI(embedding_col=column + '_emb_ada',
                                                     sentences_col='sentences',
                                                     openai_organization=openai_organization,
                                                     openai_token=openai_token
                                                    )
            v_df = text_vectorizer.process_dataset(preprocessed_df)
            vectorized_ada_df = pd.concat([vectorized_ada_df, v_df[column + '_emb_ada']], axis=1)
    # drop duplicate columns (keep first by default)
    vectorized_ada_df = vectorized_ada_df.loc[:,~vectorized_ada_df.columns.duplicated()]
    # Save vectorized dataset to pickle
    with open(f'./models/{model_path}/{openai_model}/data/vectorized_{tag}_dataset.pkl', 'wb') as f:
        pickle.dump(vectorized_ada_df, f)
else:
    # Load vectorized dataset from pickle
    with open(f'./models/{model_path}/{openai_model}/data/vectorized_{tag}_dataset.pkl', 'rb') as f:
        vectorized_ada_df = pickle.load(f)
"""
Calculate similarity matrices from the vectorized dataset
The columns present in the dataframes specified below represent combinations of text representation and embedding
    - full_text_emb_hs: full text vectorized using the last hidden states from the selected pre-trained model
    - full_text_emb_if: full text vectorized using the inference API from the selected pre-trained model
    - full_text_emb_ada: full text vectorized using the embedding API from Open AI (openai_model)

if model_path = 'dominguesm/legal-bert-ner-base-cased-ptbr':
    - summarized_text_O_emb_hs: summarized text removing predictions in ['O'] vectorized using the last hidden states from the pre-trained model
    - summarized_text_O_TEMPO_LOCAL_emb_hs: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL'] vectorized using the last hidden states from the pre-trained model
    - summarized_text_LEG_JURIS_emb_hs: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL', 'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO'] vectorized using the last hidden states from the pre-trained model
    - summarized_text_O_emb_if: summarized text removing predictions in ['O'] vectorized using the inference API from the pre-trained model
    - summarized_text_O_TEMPO_LOCAL_emb_if: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL'] vectorized using the inference API from the pre-trained model
    - summarized_text_LEG_JURIS_emb_if: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL', 'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO'] vectorized using the inference API from the pre-trained model
    - summarized_text_O_emb_ada: summarized text removing predictions in ['O'] vectorized using the embedding API from Open AI (openai_model)
    - summarized_text_O_TEMPO_LOCAL_emb_ada: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL'] vectorized using the embedding API from Open AI (openai_model)
    - summarized_text_LEG_JURIS_emb_ada: summarized text removing predictions in ['O', 'B-TEMPO', 'I-TEMPO', 'B-LOCAL', 'I-LOCAL', 'B-PESSOA', 'I-PESSOA', 'B-ORGANIZACAO', 'I-ORGANIZACAO'] vectorized using the embedding API from Open AI (openai_model)
    
if model_path = 'lisaterumi/postagger-portuguese':
    - summarized_text_N_emb_hs: summarized text keeping predictions in ['N'] vectorized using the last hidden states from the pre-trained model
    - summarized_text_N_V_emb_hs: summarized text keeping predictions in ['N', 'V'] vectorized using the last hidden states from the pre-trained model
    - summarized_text_N_emb_if: summarized text keeping predictions in ['N'] vectorized using the inference API from the pre-trained model
    - summarized_text_N_V_emb_if: summarized text keeping predictions in ['N', 'V'] vectorized using the inference API from the pre-trained model
    - summarized_text_N_emb_ada: summarized text keeping predictions in ['N'] vectorized using the embedding API from Open AI (openai_model)
    - summarized_text_N_V_emb_ada: summarized text keeping predictions in ['N', 'V'] vectorized using the embedding API from Open AI (openai_model)
"""

subfolder = 'sim_matrices'

for df, tag, path, emb_type in zip([vectorized_hs_df, vectorized_if_df, vectorized_ada_df],
                                    ['hs', 'if', 'ada'],
                                    [model_path, model_path, f'{model_path}/{openai_model}'],
                                    ['tensors', 'tensors', 'lists']
                                    ):
    labels = summ_models[model_path]['labels_to_keep'].keys()
    columns = [f"full_text_emb_{tag}"]
    for label in labels:
        columns.append(f"summarized_text_{label}_emb_{tag}")

    print(f"Calculating similarity matrices for {tag}...")
    similarity_matrix_obj = SimilarityMatrixCalculator(df, columns, path)
    similarity_matrix_obj.calculate_similarity_matrices(emb_type)
    similarity_matrix_obj.save_similarity_matrices(subfolder=subfolder)
