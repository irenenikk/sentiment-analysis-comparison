#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from ngram_utils import get_bigram_list, get_uni_and_bi_grams, filter_least_frequent, get_fraction_of_sentiment
from naiveB import NaiveB

from science_utils import sign_test, sample_variance

def build_test_data(test_data):
    """ Build a test dataset from unigrams using the file ids. """
    labelled_test_data = test_data.groupby('file_id')['ngram'].apply(lambda gs: list(gs)).reset_index()
    labelled_test_data['review'] = labelled_test_data['ngram'].apply(lambda x: ' '.join(x))
    labelled_test_data = labelled_test_data.drop('ngram', axis=1)
    labelled_test_data['sentiment'] = labelled_test_data['file_id'].apply(lambda x: 1 if x[:3] == 'POS' else -1)
    return labelled_test_data

def estimate_accuracy(test_data, training_data_files, naive_B, smooth):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    acc = 0
    groups = 0
    for file_id, group in test_data.groupby('file_id'):
        groups += 1
        label = naive_B.predict(' '.join(group['ngram'].values))
        # make sure each file is only associated with one sentiment
        # otherwise there's a bug in reading of the data
        assert(group['sentiment'].nunique() == 1)
        acc += (label == group['sentiment'].unique()[0])
    return acc/test_data['file_id'].nunique()


def cross_validate(data, classes, folds, smooth, lowercase, unigrams=None, bigrams=None):
    file_amount = data['file_no'].nunique()
    indx = np.arange(0, file_amount, folds)
    scores = np.zeros(folds)
    for f in range(folds):
        test_data_mask = data['file_no'].isin(indx+f)
        training_file_ids = data[~test_data_mask]['file_no'].unique()
        test_data = data[test_data_mask]
        naive_B = NaiveB(classes, training_file_ids, smooth, unigrams, bigrams, lowercase)
        acc = estimate_accuracy(test_data, training_file_ids, naive_B, smooth=smooth)
        scores[f] = acc
    return np.mean(scores), sample_variance(scores)

if __name__ == '__main__':

    data_folder = 'data-tagged'

    unigrams, bigrams, = get_uni_and_bi_grams(data_folder)
    print('Got', len(unigrams), 'unigrams and', len(bigrams), 'bigrams')

    positives = get_fraction_of_sentiment(unigrams, 1)
    negatives = 1-positives
    print('Fraction of positive reviews', positives, 'and negatives', negatives)

    unigrams = filter_least_frequent(unigrams, 4)
    bigrams = filter_least_frequent(bigrams, 7)
    print('After filtering,', len(unigrams), 'unigrams and', len(bigrams), 'bigrams')

    max_training_id = 899
    single_split_train_files = list(range(max_training_id))
    
    # build the classifiers with different dataset combinations
    uni_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=False, unigrams=unigrams)
    bi_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=False, bigrams=bigrams)
    uni_bi_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=False, unigrams=unigrams, bigrams=bigrams)
    naive_bs = [uni_naiveB,
                bi_naiveB,
                uni_bi_naiveB] 
    test_data = unigrams[~unigrams['file_no'].isin(single_split_train_files)]
    labelled_test_data = build_test_data(test_data)

    print('-------------------')
    print('Naive bayes baseline (no smoothing) using the files up to', max_training_id, 'in training and the rest as a test set.')
    for model in naive_bs:
        print(model)
        print(estimate_accuracy(test_data, single_split_train_files, model, smooth=False))
        print('----')
    print('--------------------')

    print('--------------------')
    print('The p-value of the effect of smoothing using the sign test')
    smoothed_uni_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=True, unigrams=unigrams)
    p_value1 = sign_test(labelled_test_data, smoothed_uni_naiveB, uni_naiveB)
    print('The p-value of smoothing using only unigrams', p_value1)
    smoothed_bi_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=True, bigrams=bigrams)
    p_value2 = sign_test(labelled_test_data, smoothed_bi_naiveB, bi_naiveB)
    print('The p-value of smoothing using only bigrams', p_value2)
    smoothed_uni_bi_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=True, unigrams=unigrams, bigrams=bigrams)
    p_value3 = sign_test(labelled_test_data, smoothed_uni_bi_naiveB, uni_bi_naiveB)
    print('The p-value of smoothing using both uni and bigrams', p_value3)


    # check the effect of lowercasing
    smoothed_lowercased_uni_naiveB = NaiveB([-1, 1], single_split_train_files, smooth=True, unigrams=unigrams, bigrams=bigrams, lowercase=True)
    p_value4 = sign_test(labelled_test_data, smoothed_uni_naiveB, smoothed_lowercased_uni_naiveB)
    print('The p-value of lowercasing using unigrams', p_value4)
    # check the effect of using bigrams
    p_value5 = sign_test(labelled_test_data, uni_naiveB, bi_naiveB)
    print('The p-value of using bigrams compared to using unigrams', p_value5)
    p_value6 = sign_test(labelled_test_data, uni_bi_naiveB, bi_naiveB)
    print('The p-value of using both unigrams and bigrams being better than using unigrams', p_value6)
    print('---------------------')

    print('Cross validation using round robin splits')
    for smooth in [False, True]:
        print('----------------')
        print(f"When {'not' if smooth == False else ''} smoothing")
        acc_mean1, acc_var1 = cross_validate(unigrams, [-1, 1], 10, smooth=smooth, lowercase=False, unigrams=unigrams)
        print('Accuracy mean', acc_mean1, 'and variance', acc_var1, 'when using only unigrams')
        acc_mean2, acc_var2 = cross_validate(unigrams, [-1, 1], 10, smooth=smooth, lowercase=False, bigrams=bigrams)
        print('Accuracy mean', acc_mean2, 'and variance', acc_var2, 'when using only bigrams')
        acc_mean3, acc_var3 = cross_validate(unigrams, [-1, 1], 10, smooth=smooth, lowercase=False, unigrams=unigrams, bigrams=bigrams)
        print('Accuracy mean', acc_mean3, 'and variance', acc_var3, 'when using both unigrams and bigrams')

