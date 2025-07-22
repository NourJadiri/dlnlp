import re
import string

import pandas as pd

from utils.preprocessing_pipeline import PreprocessingPipeline

def tokenize(text, tokenize_punct):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text:
        return []
    if tokenize_punct:
        # Keep punctuation as separate tokens
        tokens = re.findall(r"\w+|[{}]".format(re.escape(string.punctuation)), text)
    else:
        # Split on any whitespace or punctuation character (punctuation is removed)
        tokens = re.split(r'[\s{}]+'.format(re.escape(string.punctuation)), text)
        tokens = [token for token in tokens if token]
    return tokens

def replace_numbers(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    return re.sub(r'\d+', '<NUM>', text)

def remove_stopwords(tokens, stopwords):
    if not isinstance(tokens, list):
        raise TypeError("Input must be a list of tokens.")
    if not isinstance(stopwords, set):
        raise TypeError("Stopwords must be a set.")
    return [token for token in tokens if token.lower() not in stopwords]

def remove_punctuation(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    return re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)

def lowercase_text(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    return text.lower()

def get_stopwords(column, max_freq=0.5):
    from collections import Counter
    from itertools import chain
    if not isinstance(column, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    
    n_docs = len(column)
    if n_docs == 0:
        return set()
    
    # document frequency
    doc_freq = Counter(chain.from_iterable(set(toks) for toks in column))
    cutoff = max_freq * n_docs

    drop_tokens = {tok for tok, dfreq in doc_freq.items() if dfreq > cutoff}

    return drop_tokens

def load_data_to_df(data, scores):
    df = pd.DataFrame(data={'text': data, 'score': scores})
    return df

def pre_process(
    reviews,
    tokenize_punct=False,
    lowercase=False,
    remove_punct=False,
    remove_high_freq_terms=False,
    high_freq_threshold=0.5,
    replace_numbers=False
):
    if not isinstance(reviews, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    pipeline = PreprocessingPipeline()
    
        
    if lowercase:
        pipeline.add_step(lowercase_text, input_column='text', output_column='text', active=True)
    
    if remove_punct:
        pipeline.add_step(remove_punctuation, input_column='text', output_column='text', active=True)

    if replace_numbers:
        pipeline.add_step(replace_numbers, input_column='text', output_column='text', active=True)
        
    if tokenize_punct:
        pipeline.add_step(lambda text: tokenize(text, tokenize_punct=True), input_column='text', output_column='tokens', active=True)
    else:
        pipeline.add_step(lambda text: tokenize(text, tokenize_punct=False), input_column='text', output_column='tokens', active=False)


    if remove_high_freq_terms:
        stopwords = get_stopwords(reviews['tokens'], max_freq=high_freq_threshold)
        pipeline.add_step(lambda tokens: remove_stopwords(tokens, stopwords), input_column='tokens', output_column='tokens', active=True)

    return pipeline.process(reviews)