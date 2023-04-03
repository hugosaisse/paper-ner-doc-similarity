import re
import pandas as pd
import torch
from transformers import AutoTokenizer

class TextPreprocessor:
    """
    This class preprocesses and tokenize legal text in portuguese
    regex_entities is a list of regular expression to remove entity names, phone numbers and addresses
    regex_abbreviations is a regular expression to remove abbreviations
    common_words is a list of common words to be removed before tokenization
    """
    def __init__(self, model_path, regex_entities=None, regex_abbreviations=None, common_words=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.regex_entities = regex_entities
        self.regex_abbreviations = regex_abbreviations
        self.common_words = common_words

    def preprocess_text(self, df, raw_text_column='text'):
        """
        This function preprocesses raw text in a DataFrame by removing common abbreviations and replacing very common words,
        and then tokenizing the resulting text.

        Input: DataFrame with columns 'id' and raw_text_column
        Output: DataFrame with columns 'id', raw_text_column, and 'tokens'
        """
        if self.regex_abbreviations:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._remove_common_abbreviations(x, self.regex_abbreviations))
        if self.regex_entities:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._regex_substitute(x, self.regex_entities))
        if self.common_words:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._replace_common_words(x, self.common_words))
        df[['inputs','tokens']] = df[raw_text_column].apply(lambda x: self._tokenize_text(x))
        return df

    def _remove_common_abbreviations(self, text, regex_abbreviations):
        """
        This function removes common abbreviations from text using regex.
        """
        return ' '.join([word for word in re.split(regex_abbreviations, text) if len(word) > 1])

    def _replace_common_words(self, text, common_words):
        """
        This function replaces very common words in text.
        """
        for word in common_words:
            text = re.sub(r'\b{}\b'.format(word), '', text)
        return text

    def _regex_substitute(self, text, regex_entities):
        """
        This function removes the matching strings from the text based on a list of regular expressions.
        """
        for regex in regex_entities:
            text = re.sub(regex, '', text)
        return text

    def _tokenize_text(self, text):
        """
        This function tokenizes text using AutoTokenizer.
        """

        inputs = self.tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        tokens = inputs.tokens()
        
        return pd.Series([inputs, tokens])

class TextPreprocessorChunks:
    """
    This class preprocesses and tokenize legal text in portuguese considering the need for chunking
    sentences longer than the max_length of the model.
    """
    def __init__(self, model_path, regex_entities=None, regex_abbreviations=None, common_words=None, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.regex_entities = regex_entities
        self.regex_abbreviations = regex_abbreviations
        self.common_words = common_words
        self.max_length = max_length

    def preprocess_text(self, df, raw_text_column='text'):
        """
        This function preprocesses raw text in a DataFrame by removing common abbreviations and replacing very common words,
        and then tokenizing the resulting text.

        Input: DataFrame with columns 'id' and raw_text_column
        Output: DataFrame with columns 'id', raw_text_column, and 'tokens'
        """
        if self.regex_abbreviations:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._remove_common_abbreviations(x, self.regex_abbreviations))
        if self.regex_entities:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._regex_substitute(x, self.regex_entities))
        if self.common_words:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._replace_common_words(x, self.common_words))
        df[['inputs', 'tokens']] = df[raw_text_column].apply(lambda x: self._tokenize_text(x))
        return df

    def _remove_common_abbreviations(self, text, regex_abbreviations):
        """
        This function removes common abbreviations from text using regex.
        """
        return ' '.join([word for word in re.split(regex_abbreviations, text) if len(word) > 1])

    def _replace_common_words(self, text, common_words):
        """
        This function replaces very common words in text.
        """
        for word in common_words:
            text = re.sub(r'\b{}\b'.format(word), '', text)
        return text

    def _regex_substitute(self, text, regex_entities):
        """
        This function removes the matching strings from the text based on a list of regular expressions.
        """
        for regex in regex_entities:
            text = re.sub(regex, '', text)
        return text

    def _split_dict_chunks(self, input_dict):
        """
        This function takes in two arguments: the input dictionary input_dict and the number of chunks num_chunks that you want to split the values into. It first initializes an empty dictionary input_chunks with the same keys as the input dictionary.

        It then calculates the chunk size based on the length of the input_ids value in the input dictionary and the desired number of chunks. If the length is not evenly divisible by the number of chunks, it adds an extra chunk to ensure that all data is included.

        Finally, it iterates over the number of chunks and creates a new dictionary input_chunk containing a subset of the values from the input dictionary for the current chunk. It then appends each value to the corresponding list in the input_chunks dictionary.

        The function returns the input_chunks dictionary with each key containing a list of dictionaries, where each dictionary represents a chunk of the original input dictionary.
        """

        input_chunks = {k: [] for k in input_dict.keys()}
        token_chunks =[]
        input_ids_split = torch.split(input_dict['input_ids'], self.max_length, dim=1)

        for i in range(len(input_ids_split)):
            input_chunk = {k: torch.split(v, self.max_length, dim=1)[i] for k, v in input_dict.items()}
            for k, v in input_chunk.items():
                input_chunks[k].append(v)
            
            token_chunk = self.tokenizer.convert_ids_to_tokens(input_chunks['input_ids'][i][0])
            token_chunks.append(token_chunk)
        
        return input_chunks, token_chunks


    def _tokenize_text(self, text):
        """
        This function tokenizes text using AutoTokenizer.
        """
        inputs = self.tokenizer(text, truncation=False, return_tensors='pt')
        inputs_chunked, tokens_chunked = self._split_dict_chunks(inputs)
        
        return pd.Series([inputs_chunked, tokens_chunked])