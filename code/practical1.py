#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from ngram_utils import get_bigram_list, get_uni_and_bi_grams, filter_least_frequent, get_fraction_of_sentiment
from naiveB import NaiveB

from science_utils import sign_test_systems, sign_test_lists, sample_variance

def build_data(test_data):
    """ Build a test dataset from unigrams using the file ids. """
    labelled_test_data = test_data.groupby('file_id')['ngram'].apply(lambda gs: list(gs)).reset_index()
    labelled_test_data['review'] = labelled_test_data['ngram'].apply(lambda x: ' '.join(x))
    labelled_test_data = labelled_test_data.drop('ngram', axis=1)
    labelled_test_data['sentiment'] = labelled_test_data['file_id'].apply(lambda x: 1 if x[:3] == 'POS' else -1)
    return labelled_test_data

def estimate_naive_b_accuracy(test_data, training_data_files, naive_B, smooth):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    acc = 0
    for file_id, group in test_data.groupby('file_id'):
        label = naive_B.predict(' '.join(group['ngram'].values))
        # make sure each file is only associated with one sentiment
        # otherwise there's a bug in reading of the data
        assert(group['sentiment'].nunique() == 1)
        acc += (label == group['sentiment'].unique()[0])
    return acc/test_data['file_id'].nunique()

def cross_validate_naiveB(data, classes, folds, smooth, lowercase, unigrams=None, bigrams=None, return_raw=False):
    """ Build an cross validate an svm with given parameters, and return mean accuracy and variance. """
    file_amount = data['file_no'].nunique()
    indx = np.arange(0, file_amount, folds)
    scores = np.zeros(folds)
    for f in range(folds):
        test_data_mask = data['file_no'].isin(indx+f)
        training_file_ids = data[~test_data_mask]['file_no'].unique()
        test_data = data[test_data_mask]
        naive_B = NaiveB(classes, training_file_ids, smooth, unigrams, bigrams, lowercase)
        acc = estimate_naive_b_accuracy(test_data, training_file_ids, naive_B, smooth=smooth)
        scores[f] = acc
    if return_raw:
        return scores
    return np.mean(scores), sample_variance(scores)

def cross_validate_naive_b_smoothing_sign_test(data, classes, folds, unigrams):
    """ Test difference in two cross-validation accuracy sequences, one obtained from a smoothed and the other from an unsmoothed system. """
    systemA_accuracies = cross_validate_naiveB(data, classes, folds, smooth=True, lowercase=False, unigrams=unigrams, return_raw=True)
    systemB_accuracies = cross_validate_naiveB(data, classes, folds, smooth=False, lowercase=False, unigrams=unigrams, return_raw=True)
    return sign_test_lists(systemA_accuracies, systemB_accuracies)

def cross_validate_naive_b_bigrams_sign_test(data, classes, folds, unigrams, bigrams):
    """ Test difference in two cross-validation accuracy sequences, one obtained using unigrams and the other using bigrams. """
    systemA_accuracies = cross_validate_naiveB(data, classes, folds, smooth=False, lowercase=False, unigrams=unigrams, return_raw=True)
    systemB_accuracies = cross_validate_naiveB(data, classes, folds, smooth=False, lowercase=False, bigrams=bigrams, return_raw=True)
    return sign_test_lists(systemA_accuracies, systemB_accuracies)

def cross_validate_naive_b_unigrams_bigrams_sign_test(data, classes, folds, unigrams, bigrams):
    """ Test difference in two cross-validation accuracy sequences, one obtained using unigrams and the other using bigrams. """
    systemA_accuracies = cross_validate_naiveB(data, classes, folds, smooth=False, lowercase=False, unigrams=unigrams, return_raw=True)
    systemB_accuracies = cross_validate_naiveB(data, classes, folds, smooth=False, lowercase=False, unigrams=unigrams, bigrams=bigrams, return_raw=True)
    return sign_test_lists(systemA_accuracies, systemB_accuracies)

def cross_validate_naive_b_lowercase_sign_test(data, classes, folds, unigrams):
    """ Test difference in two cross-validation accuracy sequences, one obtained from a lowercased and the other from an unlowercased system. """
    systemA_accuracies = cross_validate_naiveB(data, classes, folds, smooth=True, lowercase=True, unigrams=unigrams, return_raw=True)
    systemB_accuracies = cross_validate_naiveB(data, classes, folds, smooth=True, lowercase=False, unigrams=unigrams, return_raw=True)
    return sign_test_lists(systemA_accuracies, systemB_accuracies)

def run_baseline_test(naive_bs, single_split_train_files, test_data):
    """ Print the accuracy of different naive bayes classifiers using a single train-test split. """
    print('Naive bayes baseline (no smoothing) using the files up to', max(single_split_train_files), 'in training and the rest as a test set.')
    for model in naive_bs:
        print(model)
        print(estimate_naive_b_accuracy(test_data, single_split_train_files, model, smooth=False))
        print('----')

def run_single_split_p_value_tests(single_split_train_files, classes, labelled_test_data, unigrams, bigrams, uni_naiveB, bi_naiveB, uni_bi_naiveB):
    """ Run different tests checking the significance of certain factors using a single train-test split. """
    print('The p-value of the effect of smoothing using the sign test with a single split')
    smoothed_uni_naiveB = NaiveB(classes, single_split_train_files, smooth=True, unigrams=unigrams)
    p_value1 = sign_test_systems(labelled_test_data, smoothed_uni_naiveB, uni_naiveB)
    print('The p-value of smoothing using only unigrams', p_value1)
    smoothed_bi_naiveB = NaiveB(classes, single_split_train_files, smooth=True, bigrams=bigrams)
    p_value2 = sign_test_systems(labelled_test_data, smoothed_bi_naiveB, bi_naiveB)
    print('The p-value of smoothing using only bigrams', p_value2)
    smoothed_uni_bi_naiveB = NaiveB(classes, single_split_train_files, smooth=True, unigrams=unigrams, bigrams=bigrams)
    p_value3 = sign_test_systems(labelled_test_data, smoothed_uni_bi_naiveB, uni_bi_naiveB)
    print('The p-value of smoothing using both uni and bigrams', p_value3)
    # check the effect of lowercasing
    smoothed_lowercased_uni_naiveB = NaiveB(classes, single_split_train_files, smooth=True, unigrams=unigrams, bigrams=bigrams, lowercase=True)
    p_value4 = sign_test_systems(labelled_test_data, smoothed_uni_naiveB, smoothed_lowercased_uni_naiveB)
    print('The p-value of lowercasing using unigrams', p_value4)
    # check the effect of using bigrams
    p_value5 = sign_test_systems(labelled_test_data, uni_naiveB, bi_naiveB)
    print('The p-value of using bigrams compared to using unigrams', p_value5)
    p_value6 = sign_test_systems(labelled_test_data, uni_bi_naiveB, bi_naiveB)
    print('The p-value of using both unigrams and bigrams being better than using unigrams', p_value6)

def run_cross_validation(classes, unigrams, bigrams):
    """ Run cross-validaion using different model parameters. """
    print('Cross validation using round robin splits')
    for smooth in [False, True]:
        print('----------------')
        print(f"When {'not' if smooth == False else ''} smoothing")
        acc_mean1, acc_var1 = cross_validate_naiveB(unigrams, classes, 10, smooth=smooth, lowercase=False, unigrams=unigrams)
        print('Accuracy mean', acc_mean1, 'and variance', acc_var1, 'when using only unigrams')
        acc_mean2, acc_var2 = cross_validate_naiveB(unigrams, classes, 10, smooth=smooth, lowercase=False, bigrams=bigrams)
        print('Accuracy mean', acc_mean2, 'and variance', acc_var2, 'when using only bigrams')
        acc_mean3, acc_var3 = cross_validate_naiveB(unigrams, classes, 10, smooth=smooth, lowercase=False, unigrams=unigrams, bigrams=bigrams)
        print('Accuracy mean', acc_mean3, 'and variance', acc_var3, 'when using both unigrams and bigrams')

def run_cross_validated_accuracy_sign_tests(classes, unigrams, bigrams, folds=10):
    """ Run sign test on accuracies obtained using cross-validation with different systems. """
    p1 = cross_validate_naive_b_smoothing_sign_test(unigrams, classes, folds, unigrams)
    print('The p-value of the effect of smoothing by cross-validated accuracies', p1)
    p2 = cross_validate_naive_b_bigrams_sign_test(unigrams, classes, folds, unigrams, bigrams)
    print('The p-value of the effect of using bigrams by cross-validated accuracies', p2)
    p3 = cross_validate_naive_b_lowercase_sign_test(unigrams, classes, folds, unigrams)
    print('The p-value of the effect of lowercasing by cross-validated accuracies', p3)
    p4 = cross_validate_naive_b_bigrams_sign_test(unigrams, classes, folds, unigrams, bigrams)
    print('The p-value of the effect of using both unigrams and bigrams by cross-validated accuracies', p4)

def get_features(data_folder):
    """ Load and filter unigrams and bigrams. """
    unigrams, bigrams, = get_uni_and_bi_grams(data_folder)
    print('Got', unigrams['ngram'].nunique(), 'unigrams and', bigrams['ngram'].nunique(), 'bigrams')

    positives = get_fraction_of_sentiment(unigrams, 1)
    negatives = 1-positives
    print('Fraction of positive reviews', positives, 'and negatives', negatives)

    unigrams = filter_least_frequent(unigrams, 4)
    bigrams = filter_least_frequent(bigrams, 7)
    print('After filtering', unigrams['ngram'].nunique(), 'unigrams and', bigrams['ngram'].nunique(), 'bigrams')
    return unigrams, bigrams

def main():
    data_folder = 'data-tagged'

    unigrams, bigrams = get_features(data_folder)
    max_training_id = 899
    single_split_train_files = list(range(max_training_id+1))
    classes = unigrams['sentiment'].unique()
    
    # build the classifiers with different dataset combinations
    uni_naiveB = NaiveB(classes, single_split_train_files, smooth=False, unigrams=unigrams)
    bi_naiveB = NaiveB(classes, single_split_train_files, smooth=False, bigrams=bigrams)
    uni_bi_naiveB = NaiveB(classes, single_split_train_files, smooth=False, unigrams=unigrams, bigrams=bigrams)
    naive_bs = [uni_naiveB,
                bi_naiveB,
                uni_bi_naiveB] 
    test_data = unigrams[~unigrams['file_no'].isin(single_split_train_files)]
    labelled_test_data = build_data(test_data)

    p4 = cross_validate_naive_b_bigrams_sign_test(unigrams, classes, 10, unigrams, bigrams)
    print('The p-value of the effect of using both unigrams and bigrams by cross-validated accuracies', p4)

    print('-------------------')
    run_baseline_test(naive_bs, single_split_train_files, test_data)
    print('-------------------')
    run_single_split_p_value_tests(single_split_train_files, classes, labelled_test_data, unigrams, bigrams, uni_naiveB, bi_naiveB, uni_bi_naiveB)
    print('---------------------')
    run_cross_validation(classes, unigrams, bigrams)
    print('--------------------')
    run_cross_validated_accuracy_sign_tests(classes, unigrams, bigrams, folds=10)

if __name__ == '__main__':
    main()