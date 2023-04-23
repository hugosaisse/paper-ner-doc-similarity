import requests
import time
import torch
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

    def __init__(self, model_path, max_length=512, embedding_col='sentence_embedding', inputs_col='inputs_chunked', device=torch.device("cpu")):
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

    def __init__(self, model_path, max_length=512, embedding_col='sentence_embedding', tokens_col='tokens_chunked', api_key=None):
        self.model_path = model_path
        self.max_length = max_length
        self.embedding_col = embedding_col
        self.tokens_col = tokens_col
        self.api_key = api_key

    def _vectorize_chunks(self, text):
        """
        This function vectorizes the tokenized chunks using the Hugging Face's Inference API and returns the
        averaged sentence embedding.

        The output_hidden_states variable in the previous code is an option that you pass to the Hugging Face Inference API to request the model to return the hidden states for all layers of the transformer model.

        By setting "output_hidden_states": True, you're asking the Inference API to include the hidden states in the response. These hidden states can be used to generate token-level embeddings or sentence-level embeddings using a pooling strategy.

        In the VectorizerInferenceAPI class, the hidden states from the last layer are used to create token-level embeddings. These embeddings are then averaged to create a sentence-level embedding, following the average pooling strategy.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        embeddings = []

        for chunk in text:
            data = {
                "inputs": chunk,
            }

            success = False
            while not success:
                response = requests.post(f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_path}",
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
            for token in output:
                hidden_state = torch.tensor(token[-1][1:-1])
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

    def process_dataset(self, dataset):
        """
        This function processes the dataset applying the vectorization function using the pandas apply() function.
        """
        dataset[self.embedding_col] = dataset[self.tokens_col].apply(self._vectorize_chunks)
        return dataset
