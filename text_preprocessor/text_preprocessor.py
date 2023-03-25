import re
import pandas as pd
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
        #df['inputs'], df['tokens'] = self._tokenize_text(df[raw_text_column])
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
