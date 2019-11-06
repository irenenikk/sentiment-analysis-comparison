#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from ngram_utils import get_bigram_list, get_uni_and_bi_grams
from naiveB import NaiveB

from science_utils import sign_test, sample_variance

def build_test_data(test_data):
    """ Build a test dataset from unigrams using the file ids. """
    labelled_test_data = test_data.groupby('file_id')['ngram'].apply(lambda gs: list(gs)).reset_index()
    labelled_test_data['review'] = labelled_test_data['ngram'].apply(lambda x: ' '.join(x[0]))
    labelled_test_data = labelled_test_data.drop('ngram', axis=1)
    labelled_test_data['sentiment'] = labelled_test_data['file_id'].apply(lambda x: 1 if x[:3] == 'POS' else -1)
    return labelled_test_data

def estimate_accuracy(test_data, training_data_files, naive_B, smooth):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    acc = 0
    groups = 0
    for file_id, group in test_data.groupby('file_id'):
        groups += 1
        label = naive_B.predict(' '.join(group['ngram'].values), training_data_files, smooth=smooth)
        # make sure each file is only associated with one sentiment
        # otherwise there's a bug in reading of the data
        assert(group['sentiment'].nunique() == 1)
        acc += (label == group['sentiment'].unique()[0])
    return acc/test_data['file_id'].nunique()


def cross_validate(naive_B, data, folds, smooth):
    file_amount = data['file_no'].nunique()
    indx = np.arange(0, file_amount, folds)
    scores = np.zeros(folds)
    for f in range(folds):
        test_data_mask = data['file_no'].isin(indx+f)
        training_file_ids = data[~test_data_mask]['file_no'].unique()
        test_data = data[test_data_mask]
        acc = estimate_accuracy(test_data, training_file_ids, naive_B, smooth=smooth)
        print('Cross validation accuracy in fold', f, ':', acc)
        scores[f] = acc
    return np.mean(scores), sample_variance(scores)

if __name__ == '__main__':

    data_folder = 'data-tagged'

    unigrams, bigrams, = get_uni_and_bi_grams(data_folder)
    print('Got', len(unigrams), 'unigrams and', len(bigrams), 'bigrams')

    # build the classifiers with different dataset combinations
    uni_naiveB = NaiveB([-1, 1], unigrams=unigrams)
    bi_naiveB = NaiveB([-1, 1], bigrams=bigrams)
    uni_bi_naiveB = NaiveB([-1, 1], unigrams=unigrams, bigrams=bigrams)
    naive_bs = [uni_naiveB,
                bi_naiveB,
                uni_bi_naiveB] 

    max_training_id = 899
    single_split_train_files = list(range(max_training_id))
    test_data = unigrams[~unigrams['file_no'].isin(single_split_train_files)]
    labelled_test_data = build_test_data(test_data)
    
    '''
    print('-------------------')
    print('Naive bayes baseline (no smoothing) using the files up to', max_training_id, 'in training and the rest as a test set.')
    for model in naive_bs:
        print(model)
        print(estimate_accuracy(test_data, single_split_train_files, model, smooth=False))
        print('----')
    print('--------------------')
    
    print('--------------------')
    print('The p-value of the effect of smoothing using the sign test')
    p_value1 = sign_test(labelled_test_data, 
                        lambda data: uni_naiveB.predict(data, list(range(899)), smooth=True), 
                        lambda data: uni_naiveB.predict(data, list(range(899)), smooth=False))
    print('The p-value of smoothing being better using only unigrams', p_value1)
    print('---')
    p_value2 = sign_test(labelled_test_data, 
                        lambda data: bi_naiveB.predict(data, list(range(899)), smooth=True), 
                        lambda data: bi_naiveB.predict(data, list(range(899)), smooth=False))
    print('The p-value of smoothing being better using only biigrams', p_value2)
    print('---')
    p_value3 = sign_test(labelled_test_data, 
                        lambda data: uni_bi_naiveB.predict(data, list(range(899)), smooth=True), 
                        lambda data: uni_bi_naiveB.predict(data, list(range(899)), smooth=False))
    print('The p-value of smoothing being better using both uni and bigrams', p_value3)
    '''

    print('Cross validation using round robin splits')
    for smooth in [True]:
        for model in naive_bs:
            print('For model', model, f"with{'out' if smooth == False else ''} smoothing")
            acc_mean, acc_var = cross_validate(model, unigrams, 10, smooth=smooth)
            print('Accuracy mean', acc_mean, 'and variance', acc_var)





