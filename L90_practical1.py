#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from scipy.stats import binom
from functools import reduce
import math


# ## Load data

# In[13]:


def get_bigram_list(tokens, min_freq=7):
    """ Find bigrams from given text and return in a pandas dataframe. """
    bigrams = []
    for i in range(1, len(tokens)):
        bigram = f'{tokens[i-1]} {tokens[i]}'
        bigrams += [bigram]
    return bigrams


# In[193]:


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
            # find unigrams
            new_unigrams = pd.read_csv(os.path.join(review_folder, file), sep='\t', header=None, names=['ngram', 'pos']).values[:,0]
            unigrams += list(new_unigrams)
            unigram_sentiments += [(1 if sent == 'POS' else -1)]*len(new_unigrams)
            unigram_file_ids += [f'{sent}-{file[2:5]}']*len(new_unigrams)
            unigram_file_nos += [int(file[2:5])]*len(new_unigrams)
            # find bigrams
            new_bigrams = get_bigram_list(new_unigrams)
            bigrams += new_bigrams            
            bigram_sentiments += [(1 if sent == 'POS' else -1)]*len(new_bigrams)
            bigram_file_ids += [f'{sent}-{file[2:5]}']*len(new_bigrams)
            bigram_file_nos += [int(file[2:5])]*len(new_bigrams)
    unigram_df = pd.DataFrame(list(zip(unigrams, unigram_sentiments, unigram_file_ids, unigram_file_nos)), columns=['ngram', 'sentiment', 'file_id', 'file_no'])
    bigram_df = pd.DataFrame(list(zip(bigrams, bigram_sentiments, bigram_file_ids, bigram_file_nos)), columns=['ngram', 'sentiment', 'file_id', 'file_no'])
    return unigram_df, bigram_df





# ## Helpers for defining the necessary probabilities


def unigram_tokenize(content):
    """ Split into unigrams by punctuation and whitespace, then lowercase and remove trailing whitespace"""
    return np.asarray(list(filter(None,((map(lambda x: x, map(str.strip, re.split('(\W)', content))))))))

def bigram_tokenize(content):
    """ Split text into bigrams """
    tokens = unigram_tokenize(content)
    for i in range(1, len(tokens)):
        yield f'{tokens[i-1]} {tokens[i]}'

def preprocess_ngrams(ngrams, sent, min_count, smooth):
    counts = ngrams[ngrams['sentiment'] == sent]['ngram'].value_counts()
    filtered = counts[counts >= min_count]
    voc_size = len(counts)
    if smooth:
        return (filtered+1)/(sum(filtered)+ngrams['ngram'].nunique())
    return filtered/sum(filtered)

def get_class_probabilites(text, classes, tokenize, data, smooth, min_freq):
    class_probs = np.zeros(len(classes))
    for i, cl in enumerate(classes):
        p = 0
        conditioned_counts = preprocess_ngrams(data, cl, min_freq, smooth)
        smooth_denom = (data['sentiment']==cl).sum()+data['ngram'].nunique()
        for word in tokenize(text):
            if word in conditioned_counts.index:
                p += np.log(conditioned_counts.loc[word])
            # apply smoothing separately if word not present in class
            elif smooth:
                p += np.log(1/(smooth_denom))
        # the prior is the fraction of documents in a specific class
        sentiment_files = data[['file_id', 'sentiment']].groupby('file_id').mean()
        prior = (sentiment_files['sentiment'] == cl).sum()/len(sentiment_files)
        p += np.log(prior)
        class_probs[i] = p
    return class_probs

def naive_binary_bayes(text, unigrams=None, bigrams=None, smooth=True):
    """ Predict the class of a string given unigrams and bigrams. """
    if unigrams is None and bigrams is None:
        raise ValueError('Please choose to use either unigrams or bigrams by providing the data')
    # set the binary classification labels
    classes = [-1, 1]
    class_probs = np.zeros(len(classes))
    if unigrams is not None:
        class_probs += get_class_probabilites(text, classes, unigram_tokenize, unigrams, smooth, min_freq=4)
    if bigrams is not None:
        class_probs += get_class_probabilites(text, classes, bigram_tokenize, bigrams, smooth, min_freq=7)
    return classes[np.argmax(class_probs)]

class NaiveB:
    
    def preprocess_ngrams(ngrams, sent, min_count, smooth):
        counts = ngrams[ngrams['sentiment'] == sent]['ngram'].value_counts()
        filtered = counts[counts >= min_count]
        voc_size = len(counts)
        if smooth:
            return (filtered+1)/(sum(filtered)+ngrams['ngram'].nunique())
        return filtered/sum(filtered)    
    
    def __init__(self, classes, unigrams=None, bigrams=None):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.classes = classes
        
    def get_class_probabilites(self, text, tokenize, data, smooth, min_freq):
        class_probs = np.zeros(len(self.classes))
        for i, cl in enumerate(self.classes):
            p = 0
            conditioned_counts = preprocess_ngrams(data, cl, min_freq, smooth)
            smooth_denom = (data['sentiment']==cl).sum()+data['ngram'].nunique()
            for word in tokenize(text):
                if word in conditioned_counts.index:
                    p += np.log(conditioned_counts.loc[word])
                # apply smoothing separately if word not present in class
                elif smooth:
                    p += np.log(1/(smooth_denom))
            # the prior is the fraction of documents in a specific class
            sentiment_files = data[['file_id', 'sentiment']].groupby('file_id').mean()
            prior = (sentiment_files['sentiment'] == cl).sum()/len(sentiment_files)
            p += np.log(prior)
            class_probs[i] = p
        return class_probs    
    
    def predict2(self, text, smooth=True):
        # set the binary classification labels
        class_probs = np.zeros(len(self.classes))
        if self.unigrams is not None:
            class_probs += self.get_class_probabilites(text, unigram_tokenize, self.unigrams, smooth, min_freq=4)
        if self.bigrams is not None:
            class_probs += self.get_class_probabilites(text, bigram_tokenize, self.bigrams, smooth, min_freq=7)
        return self.classes[np.argmax(class_probs)]
    
    def predict(self, text, training_data_files, smooth=True):
        # set the binary classification labels
        class_probs = np.zeros(len(self.classes))
        if self.unigrams is not None:
            class_probs += self.get_class_probabilites(text, unigram_tokenize, self.unigrams[self.unigrams['file_no'].isin(training_data_files)], smooth, min_freq=4)
        if self.bigrams is not None:
            class_probs += self.get_class_probabilites(text, bigram_tokenize, self.bigrams[self.bigrams['file_no'].isin(training_data_files)], smooth, min_freq=7)
        return self.classes[np.argmax(class_probs)]    


# In[230]:
def estimate_accuracy2(test_data, unigrams=None, bigrams=None):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    acc = 0
    for file_id, group in test_data.groupby('file_id'):
        label = naive_binary_bayes(' '.join(group['ngram'].values), smooth=smooth, unigrams=unigrams, bigrams=bigrams)
        # make sure each file is only associated with one sentiment
        # otherwise there's a bug in reading of the data
        assert(group['sentiment'].nunique() == 1)
        acc += (label == group['sentiment'].unique()[0])
    return acc/test_data['file_id'].nunique()

def estimate_accuracy(test_data, training_data_files, naive_B, smooth):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    acc = 0
    for file_id, group in test_data.groupby('file_id'):
        label = naive_B.predict(' '.join(group['ngram'].values), training_data_files, smooth=smooth)
        # make sure each file is only associated with one sentiment
        # otherwise there's a bug in reading of the data
        assert(group['sentiment'].nunique() == 1)
        acc += (label == group['sentiment'].unique()[0])
    return acc/test_data['file_id'].nunique()

def calculate_p_value(N, k, q):
    res = 0
    for i in range(k):
        res += binom.pmf(i, N, q)
    return 2*res

def sign_test(data, system_A, system_B, n=10):
    plus, minus, null = 0, 0, 0
    for i in range(n):
        print('test', i+1, 'out of', n)
        a = system_A(data)
        b = system_B(data)
        if a > b:
            plus += 1
        elif a < b:
            minus += 1
        else:
            null += 1
    N = 2*math.ceil(null/2)+plus+minus
    k = math.ceil(null/2)+min(plus, minus)
    return calculate_p_value(N, k, 0.5)

def smoothed_unigram_bayes(data):
    uni_naiveB = NaiveB([-1, 1], unigrams=unigrams)
    return estimate_accuracy(data, unigrams=training_unigrams, smooth=True)

def unsmoothed_unigram_bayes(data):
    return estimate_accuracy(data, unigrams=training_unigrams, smooth=False)

def cross_validate(naive_B, data, folds):
    file_amount = data['file_no'].nunique()
    indx = np.arange(0, file_amount, folds)
    cumu_score = 0
    for f in range(folds):
        print(indx+f)
        test_data_mask = data['file_no'].isin(indx+f)
        training_file_ids = data[~test_data_mask]['file_no'].unique()
        test_data = data[test_data_mask]
        acc = estimate_accuracy(test_data, training_file_ids, naive_B, smooth=True)
        print(acc)
        cumu_score += acc
    return cumu_score/folds


# In[264]:

unigrams, bigrams, = get_uni_and_bi_grams('data-tagged')
uni_naiveB = NaiveB([-1, 1], unigrams=unigrams)
#bi_naiveB = NaiveB([-1, 1], bigrams=bigrams)
#uni_bi_naiveB = NaiveB([-1, 1], unigrams=unigrams, bigrams=bigrams)

cross_validate(uni_naiveB, unigrams, 10)



