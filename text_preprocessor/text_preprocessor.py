import re
import pandas as pd
import torch
from transformers import AutoTokenizer

class TextPreprocessor:
    """
    This class preprocesses and tokenize legal text in portuguese
    regex_list is a list of regular expression to remove entity names, phone numbers and addresses
    regex_abbreviations is a regular expression to remove abbreviations
    common_words is a list of common words to be removed before tokenization
    """
    def __init__(self, model_path, regex_list=None, regex_abbreviations=None, common_words=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.regex_list = regex_list
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
        if self.regex_list:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._regex_substitute(x, self.regex_list))
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

    def _regex_substitute(self, text, regex_list):
        """
        This function removes the matching strings from the text based on a list of regular expressions.
        """
        for regex in regex_list:
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
    def __init__(self,
                 model_path,
                 regex_list=None,
                 regex_abbreviations=None,
                 common_words=None,
                 max_length=512,
                 use_strided_chunks=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.regex_list = regex_list
        self.regex_abbreviations = regex_abbreviations
        self.common_words = common_words
        self.max_length = max_length
        self.use_strided_chunks = use_strided_chunks

    def preprocess_text(self, df, raw_text_column='text'):
        """
        This function preprocesses raw text in a DataFrame by removing common abbreviations and replacing very common words,
        and then tokenizing the resulting text.

        Input: DataFrame with columns 'id' and raw_text_column
        Output: DataFrame with columns 'id', raw_text_column, and 'tokens'
        """
        if self.regex_abbreviations:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._remove_common_abbreviations(x, self.regex_abbreviations))
        if self.regex_list:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._regex_substitute(x, self.regex_list))
        if self.common_words:
            df[raw_text_column] = df[raw_text_column].apply(lambda x: self._replace_common_words(x, self.common_words))
        df[['inputs', 'tokens', 'sentences']] = df[raw_text_column].apply(lambda x: self._tokenize_text(x))
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

    def _regex_substitute(self, text, regex_list):
        """
        This function removes the matching strings from the text based on a list of regular expressions.
        """
        for regex in regex_list:
            text = re.sub(regex, '', text)
        return text

    def _split_dict_chunks(self, input_dict):
        """
        This function takes in the input dictionary input_dict and splits the values into chunks.
        It first initializes an empty dictionary input_chunks with the same keys as the input dictionary.
        It then calculates the chunk size based on the length of the input_ids value in the input dictionary.
        Finally, it iterates over the number of chunks and creates a new dictionary input_chunk containing a
        subset of the values from the input dictionary for the current chunk. It then appends each value to the
        corresponding list in the input_chunks dictionary.
        The function returns the input_chunks dictionary with each key containing a list of dictionaries,
        where each dictionary represents a chunk of the original input dictionary.
        """

        input_chunks = {k: [] for k in input_dict.keys()}
        token_chunks = []
        sentence_chunks = []

        if self.use_strided_chunks:
            stride = self.max_length // 2
            input_ids = input_dict['input_ids']
            n = input_ids.size(1)

            # Create strided chunks
            idx_start = 0
            while idx_start < n:
                idx_end = min(idx_start + self.max_length, n)
                chunk_slice = slice(idx_start, idx_end)

                input_chunk = {k: v[:, chunk_slice] for k, v in input_dict.items()}
                for k, v in input_chunk.items():
                    input_chunks[k].append(v)

                token_chunk = self.tokenizer.convert_ids_to_tokens(input_chunks['input_ids'][-1][0],
                                                                   skip_special_tokens=True)
                token_chunks.append(token_chunk)

                # Convert a list of lists of token ids into a list of strings
                sentence_chunk = self.tokenizer.decode(input_chunks['input_ids'][i][0],
                                                       #skip_special_tokens=True,
                                                       #clean_up_tokenization_spaces=True
                                                    )
                sentence_chunks.append(sentence_chunk)

                idx_start += stride
        else:
            input_ids_split = torch.split(input_dict['input_ids'], self.max_length, dim=1)

            for i in range(len(input_ids_split)):
                input_chunk = {k: torch.split(v, self.max_length, dim=1)[i] for k, v in input_dict.items()}
                for k, v in input_chunk.items():
                    input_chunks[k].append(v)
                
                token_chunk = self.tokenizer.convert_ids_to_tokens(input_chunks['input_ids'][i][0],
                                                                   #skip_special_tokens=True
                                                                   )
                token_chunks.append(token_chunk)

                # Convert a list of lists of token ids into a list of strings
                sentence_chunk = self.tokenizer.decode(input_chunks['input_ids'][i][0],
                                                       #skip_special_tokens=True,
                                                       #clean_up_tokenization_spaces=True
                                                       )
                sentence_chunks.append(sentence_chunk)
        
        return input_chunks, token_chunks, sentence_chunks


    def _tokenize_text(self, text):
        """
        This function tokenizes text using AutoTokenizer.
        """
        inputs = self.tokenizer(text, truncation=False, return_tensors='pt')
        inputs_chunked, tokens_chunked, sentences_chunked = self._split_dict_chunks(inputs)
        
        return pd.Series([inputs_chunked, tokens_chunked, sentences_chunked])