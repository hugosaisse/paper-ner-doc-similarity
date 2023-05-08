import numpy as np
import openai
from openai.embeddings_utils import get_embedding
import os
import pickle
import requests
import tiktoken
import time
import torch
from tqdm import tqdm
from transformers import AutoModel

"""
    The classes below use transformers to produce text embeddings (vectorization)
    according to the following strategies:

    1) Feature extraction (Hidden States):
        We use the hidden states as features and just train a classifier on them, without
        modifying the pretrained model.
        + dominguesm/legal-bert-base-cased-ptbr, DistilBERT
        + Fixed window equal to max_length, average pooling, max pooling, or concatenation
        + Sliding window with a stride, average pooling, max pooling, or concatenation

    2) Hugging Face inference API:
        We use the Hugging Face inference API to generate embeddings for each token
        + dominguesm/legal-bert-base-cased-ptbr, DistilBERT
        + Fixed window equal to max_length, average pooling, max pooling, or concatenation
        + Sliding window with a stride, average pooling, max pooling, or concatenation
    
    With the last hidden states approach, the entire input sequence is passed through
    the transformer model to generate a single vector representation. With the approach
    used by the Hugging Face inference API, each token in the input sequence is passed
    through the transformer model to generate a corresponding embedding, and these
    embeddings are then combined using a pooling strategy.

    The advantage of the Hugging Face approach is that it can capture more fine-grained
    information about the input text than the last hidden state approach, since it
    generates embeddings for each token in the input sequence. However, it is also
    more computationally expensive than the last hidden state approach, since it involves
    passing each token through the transformer model separately.

    Both the Hugging Face inference API and the technique of using the last hidden
    states from a pre-trained transformer model are limited by the maximum length
    of the model's input sequence.
    
"""

class VectorizerHiddenStates:
    """
    This class takes the tokenized chunks from TextPreprocessorChunks and vectorizes them using a pre-trained
    model. It then averages the embeddings to obtain the sentence embedding.
    """

    def __init__(self, model_path, max_length=512, embedding_col='sentence_embedding', inputs_col='inputs', device=torch.device("cpu")):
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.max_length = max_length
        self.embedding_col = embedding_col
        self.inputs_col = inputs_col
        self.device = device

    def _vectorize_chunks(self, text):
        """
        This function vectorizes the tokenized chunks using the pre-trained model and returns the
        averaged sentence embedding.
        """
        with torch.no_grad():
            embeddings = [self.model(chunk.unsqueeze(0).to(self.device)).last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze(0).cpu() for chunks in text['input_ids'] for chunk in chunks]
            sentence_embedding = torch.stack(embeddings).mean(dim=0)

        return sentence_embedding

    def process_dataset(self, dataset):
        """
        This function processes the dataset applying the vectorization function using the pandas apply() function.
        """
        dataset[self.embedding_col] = dataset[self.inputs_col].apply(self._vectorize_chunks)
        return dataset

class VectorizerInferenceAPI:
    """
    This class uses Hugging Face's Inference API to get embeddings from a pre-trained model.
    It then averages the embeddings to obtain the sentence embedding.
    """

    def __init__(self, embedding_col='sentence_embedding', sentences_col='sentences', hf_token=None, api_url=None):
        self.embedding_col = embedding_col
        self.sentences_col = sentences_col
        self.hf_token = hf_token
        self.api_url = api_url

    def _vectorize_chunks(self, text):
        """
        This function vectorizes the tokenized chunks using the Hugging Face's Inference API and returns the
        averaged sentence embedding.

        The output_hidden_states variable is an option that you pass to the Hugging Face Inference API to request the model to return the hidden states for all layers of the transformer model.

        By setting "output_hidden_states": True, you're asking the Inference API to include the hidden states in the response. These hidden states can be used to generate token-level embeddings
        or sentence-level embeddings using a pooling strategy.

        In the VectorizerInferenceAPI class, the hidden states from the last layer are used to create token-level embeddings. These embeddings are then averaged to create a sentence-level embedding, following the average pooling strategy.
        """
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        embeddings = []

        for chunk in tqdm(text):
            data = {
                "inputs": chunk,
            }

            success = False
            while not success:
                response = requests.post(self.api_url,
                                        headers=headers,
                                        json=data)

                if response.status_code == 200:
                    success = True
                elif response.status_code == 503:
                    time.sleep(5)
                elif response.status_code != 503:
                    raise ValueError(f"Inference API returned a non-200 status code: {response.status_code}")

            output = response.json()

            token_embeddings = []
            for chunk in output:
                hidden_state = torch.tensor(chunk)
                """
                Some token embeddings result in embeddings with size different from [1, 768], such as [3, 768]
                It happens because when you pass a chunk of text to the feature-extraction pipeline,
                the tokenizer splits the text into subword tokens to generate embeddings.
                Some words may be tokenized into multiple subword tokens, which is why you are seeing variable tensor sizes for different tokens.
                """
                # Check the dimensions and average the subword embeddings if necessary
                if hidden_state.size(0) != 1:
                    hidden_state = hidden_state.mean(dim=0, keepdim=True)
                token_embeddings.append(hidden_state)
            
            # Average subword token embeddings to get a single token embedding
            chunk_embedding = torch.stack(token_embeddings).mean(dim=0)
            embeddings.append(chunk_embedding)

        try:
            sentence_embedding = torch.stack(embeddings).mean(dim=0)
        except RuntimeError:
            sentence_embedding = torch.zeros((1, 768))
        return sentence_embedding

    def save_temp_dataframe(self, dataset, temp_filename='temp_dataframe.pickle'):
        """
        Saves the temporary DataFrame to a pickle file.
        """
        with open(temp_filename, 'wb') as f:
             pickle.dump(dataset, f)

    def load_temp_dataframe(self, temp_filename='temp_dataframe.pickle'):
        """
        Loads the temporary DataFrame from a pickle file.
        """
        if os.path.exists(temp_filename):
            with open(temp_filename, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        else:
            return None

    def process_dataset(self, dataset):
        """
        This function processes the dataset applying the vectorization function row by row.
        It saves a temporary DataFrame after each row is processed.
        """
        temp_dataset = self.load_temp_dataframe()

        if temp_dataset is not None:
            if self.embedding_col not in temp_dataset.columns:
                temp_dataset[self.embedding_col] = None
            start_idx = temp_dataset[self.embedding_col].notna().sum()
            temp_dataset[self.sentences_col] = dataset[self.sentences_col]

        else:
            start_idx = 0
            temp_dataset = dataset.copy(deep=True)
            temp_dataset[self.embedding_col] = None

        for i in range(start_idx, len(dataset)):
            temp_dataset.at[i, self.embedding_col] = self._vectorize_chunks(temp_dataset.at[i, self.sentences_col])
            # save the temporary dataframe every 5 rows or when the last row is processed
            if (np.mod(i, 5) == 0) or (i == len(dataset) - 1):
                self.save_temp_dataframe(temp_dataset)
                print(f'Saved temporary DataFrame column {self.embedding_col} after processing row {i}.')

        return temp_dataset

class VectorizerEmbeddingOpenAI:
    """
    This class uses Chat GPT's embedding API to get embeddings from a pre-trained model.
    It then averages the embeddings to obtain the sentence embedding.
    """

    def __init__(self,
                 embedding_model='text-embedding-ada-002',
                 embedding_encoding='cl100k_base',
                 max_tokens = 8191, # 8191 is the maximum number of tokens allowed by the text-embedding-ada-002 model
                 embedding_col='sentence_embedding',
                 sentences_col='sentences',
                 openai_organization=None,
                 openai_token=None,
                 api_url=None):
        self.embedding_model = embedding_model
        self.embedding_encoding = embedding_encoding
        self.max_tokens = max_tokens
        self.embedding_col = embedding_col
        self.sentences_col = sentences_col
        self.openai_organization = openai_organization
        self.openai_token = openai_token
        self.api_url = api_url

    def _token_limiter(self, text):
        """
        Counts the number of tokens in a text chunk and limits the number of tokens to the maximum allowed by the model.
        This is to limit the number of tokens when creating the request to the API.
        """
        encoding = tiktoken.get_encoding(self.embedding_encoding)
        num_tokens = len(encoding.encode(text))
        while num_tokens > self.max_tokens:
            text = " ".join(text.split()[:-1])
            num_tokens = len(encoding.encode(text))

        return text
    
    def _vectorize_chunks(self, text):
        """
        This function vectorizes a text chunk using the OpenAI Embeddings API.
        https://platform.openai.com/docs/api-reference/embeddings

        The embedding size of the text-embedding-ada-002 model is 1536.
        If the text chunk is empty, this function returns a vector of 1536 zeros.

        The API allows a maximum of 8192 tokens per request.
        Hence we use _token_limiter to limit the number of tokens to 8192.
        """
        #text = " ".join(text.split()[:self.max_tokens])
        if len(text) == 0:
            embedding = [0]*1536
        else:
            # limit the number of words to max_tokens
            text = self._token_limiter(text)
            embedding = get_embedding(text, engine=self.embedding_model)

        return embedding

    def save_temp_dataframe(self, dataset, temp_filename='temp_dataframe_ada.pickle'):
        """
        Saves the temporary DataFrame to a pickle file.
        """
        with open(temp_filename, 'wb') as f:
             pickle.dump(dataset, f)

    def load_temp_dataframe(self, temp_filename='temp_dataframe_ada.pickle'):
        """
        Loads the temporary DataFrame from a pickle file.
        """
        if os.path.exists(temp_filename):
            with open(temp_filename, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        else:
            return None

    def process_dataset(self, dataset):
        """
        This function processes the dataset applying the vectorization function using the pandas apply() function.
        """
        openai.api_key = self.openai_token
        openai.organization = self.openai_organization
        temp_dataset = self.load_temp_dataframe()

        if temp_dataset is not None:
            if self.embedding_col not in temp_dataset.columns:
                temp_dataset[self.embedding_col] = None
            start_idx = temp_dataset[self.embedding_col].notna().sum()
            temp_dataset[self.sentences_col] = dataset[self.sentences_col]

        else:
            start_idx = 0
            temp_dataset = dataset.copy(deep=True)
            temp_dataset[self.embedding_col] = None

        print(f'Starting from row {start_idx}.')
        for i in tqdm(range(start_idx, len(dataset))):
            temp_dataset.at[i, self.embedding_col] = self._vectorize_chunks(temp_dataset.at[i, self.sentences_col])
            # save the temporary dataframe every 10 rows or when the last row is processed
            if (np.mod(i, 10) == 0) or (i == len(dataset) - 1):
                self.save_temp_dataframe(temp_dataset)
                print(f'Saved temporary DataFrame column {self.embedding_col} after processing row {i}.')

        return temp_dataset
        
        
        #dataset[self.embedding_col] = dataset[self.sentences_col].progress_apply(self._vectorize_chunks)
        
        
        #return dataset
    