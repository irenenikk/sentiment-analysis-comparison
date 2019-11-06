import os
import pandas as pd
import numpy as np
import re

def get_ngram_probabilities(ngrams, sent, smooth):
    """ 
        Count the frequency of each unigram in a given sentiment. 
        Returns the conditional probability of each ngram given a sentiment, possibly applying smoothing.
    """
    counts = ngrams[ngrams['sentiment'] == sent]['ngram'].value_counts()
    if smooth:
        return (counts+1)/(sum(counts)+len(counts))
    return counts/sum(counts)

def filter_least_frequent(data, min_freq):
    counts = data['ngram'].value_counts()
    filtered = counts[counts >= min_freq]
    return data[data['ngram'].isin(filtered.index)]

def get_bigram_list(tokens):
    """ Find bigrams from given text and return in a pandas dataframe. """
    bigrams = []
    for i in range(1, len(tokens)):
        bigram = f'{tokens[i-1]} {tokens[i]}'
        bigrams += [bigram]
    return bigrams

def get_uni_and_bi_grams(data_folder):
    """ Find unigrams and bigrams in given text and return in a pandas dataframe. """
    sentiments = ['POS', 'NEG']    
    # use lists to avoid calling append in a loop
    # unigrams
    unigrams = []
    unigram_sentiments = []
    unigram_file_ids = []
    unigram_file_nos = []
    # bigrams
    bigrams = []
    bigram_sentiments = []
    bigram_file_ids = []
    bigram_file_nos = []    
    for sent in sentiments:
        review_folder = f'{data_folder}/{sent}'
        for file in os.listdir(review_folder):
            sentiment = (1 if sent == 'POS' else -1)
            file_id = f'{sent}-{file[2:5]}'
            file_no = int(file[2:5])
            # find unigrams
            new_unigrams = pd.read_csv(os.path.join(review_folder, file), sep='\t', header=None, names=['ngram', 'pos']).values[:,0]
            unigrams += list(new_unigrams)
            unigram_sentiments += [sentiment]*len(new_unigrams)
            unigram_file_ids += [file_id]*len(new_unigrams)
            unigram_file_nos += [file_no]*len(new_unigrams)
            # find bigrams
            new_bigrams = get_bigram_list(new_unigrams)
            bigrams += new_bigrams            
            bigram_sentiments += [sentiment]*len(new_bigrams)
            bigram_file_ids += [file_id]*len(new_bigrams)
            bigram_file_nos += [file_no]*len(new_bigrams)
    unigram_df = pd.DataFrame(list(zip(unigrams, unigram_sentiments, unigram_file_ids, unigram_file_nos)), columns=['ngram', 'sentiment', 'file_id', 'file_no'])
    bigram_df = pd.DataFrame(list(zip(bigrams, bigram_sentiments, bigram_file_ids, bigram_file_nos)), columns=['ngram', 'sentiment', 'file_id', 'file_no'])
    return unigram_df, bigram_df

def unigram_tokenize(content, lowercase=True):
    """ Split into unigrams by punctuation and whitespace, then lowercase and remove trailing whitespace"""
    tokens = list(filter(None,((map(lambda x: x, map(str.strip, re.split('(\W)', content)))))))
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return np.asarray(tokens)

def bigram_tokenize(content, lowercase=True):
    """ Split text into bigrams """
    tokens = unigram_tokenize(content)
    for i in range(1, len(tokens)):
        bigram = f'{tokens[i-1]} {tokens[i]}'
        if lowercase:
            yield bigram.lower()
        yield bigram
