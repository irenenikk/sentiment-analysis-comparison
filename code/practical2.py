import re
import gensim
from gensim.test.utils import common_texts
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import svm
from joblib import dump, load
from doc_utils import doc_tokenize, get_reviews, \
                        train_doc2vec_model, get_doc2vec_data, \
                                get_bow_vectors, evaluate_vector_qualities, model_error_analysis, \
                                    calculate_vector_distances
from science_utils import sample_variance, \
                            get_mask, get_train_test_split, \
                                train_test_split_indexes, get_accuracy, permutation_test
from ngram_utils import get_uni_and_bi_grams
import time
import numpy as np
from practical1 import build_data
from scipy import spatial
from visualisation import heat_plot_two_vectors, visualise_individual_reviews, visualize_vectors, plot_matrix
import itertools
import sys

models_file = 'models'

def build_svm_classifier(X, Y, pretrained=False, save=False, **kwargs):
    model_path = os.path.join(models_file, 'svm_shape={}_{}.joblib'.format(X.shape[0], X.shape[1]))
    # a bug: how to distinguish between classifiers run with the same amount of different data
    if pretrained and os.path.exists(model_path):
        print('Loaded a pretrained SVM from', model_path)
        return load(model_path)
    #print('Training an SVM classifier with params', kwargs)
    classifier = svm.SVC(**kwargs)
    classifier.fit(X, Y)
    if save and not os.path.exists(models_file):
        os.mkdir(models_file)
        dump(classifier, model_path)
        print('Saving to', model_path)
    return classifier

def estimate_svm_accuracy(X, Y, svm):
    """ Estimate the accuracy over given test dataset. """
    preds = svm.predict(X)
    return get_accuracy(preds, Y)

def cross_validate_svm(X, Y, folds=10, return_raw=False, **kwargs):
    """ Run n-fold cross-validation using an SVM. """
    indexes = list(range(len(X)))
    rand_perm = np.random.permutation(indexes)
    step = len(X)//folds
    scores = np.zeros(folds)
    for f in range(folds):
        indices = rand_perm[f*step:f*step+step] if f*step+step < len(rand_perm) else rand_perm[f*step:]
        test_mask = get_mask(len(X), indices)
        test_X, test_Y = X[test_mask], Y[test_mask]
        train_X, train_Y = X[~test_mask], Y[~test_mask]
        svm = build_svm_classifier(train_X, train_Y, **kwargs)
        acc = estimate_svm_accuracy(test_X, test_Y, svm)
        scores[f] = acc
    if return_raw:
        return scores
    return np.mean(scores), sample_variance(scores)

def cross_validate_permutation_test(data, folds, run_perm_test, **kwargs):
    """ Run permutation tests on cross-validation folds. """
    indexes = list(range(len(data)))
    rand_perm = np.random.permutation(indexes)
    step = len(data)//folds
    p_values = np.zeros(folds)
    for f in range(folds):
        indices = rand_perm[f*step:f*step+step] if f*step+step < len(rand_perm) else rand_perm[f*step:]
        test_mask = get_mask(len(data), indices)
        test = data[test_mask]
        train = data[~test_mask]
        p_value = run_perm_test(train, test, **kwargs)
        p_values[f] = p_value
    # get p-value for all folds
    print('p values', p_values)
    return np.mean(p_values), sample_variance(p_values)

def find_optimal_c_for_rbf(X, Y):
    """ Find optimal C and a kernel. """
    c_range = np.arange(0.1, 10, 0.2)
    maxx = -1
    best_c = -1
    for c in c_range:
            accuracy = cross_validate_svm(X, Y, folds=3, kernel='rbf', C=c, gamma='scale')
            if accuracy[0] > maxx:
                maxx = accuracy[0]
                best_c = c
    print('Best accuracy with gaussian kernel:', maxx, 'and gamma = scale, and c =', best_c)
    return best_c

def find_optimal_doc2vec_hyperparams(imdb_reviews, dev_data):
    maxim = -1
    max_params = {}
    for window_size in [12]:
        for epochs in [30]:
            for dm in [0]:
                for vec_size in [100]:
                    print('-----------------------------------------')
                    print('window size', window_size, 'epochs', epochs, 'dm', dm, 'vec size', vec_size)
                    train_X, train_Y, model_imdb = train_doc2vec_model(imdb_reviews, epochs=epochs, vec_size=vec_size, window_size=window_size, dm=dm)
                    test_X = get_doc2vec_data(dev_data['review'].values, model_imdb)
                    test_Y = dev_data['sentiment'].values
                    print('For test set vectors inferred with doc2vec')
                    svm1 = build_svm_classifier(train_X, train_Y, kernel='linear')
                    accuracy1 = estimate_svm_accuracy(test_X, test_Y, svm1)
                    print('accuracy using doc2vec and svm with a linear kernel', accuracy1)
                    svm2 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale')
                    accuracy2 = estimate_svm_accuracy(test_X, test_Y, svm2)
                    print('accuracy using doc2vec and svm with a gaussian kernel', accuracy2)
                    if accuracy1 > maxim:
                        maxim = accuracy1
                        max_params = { 'kernel': 'linear', 'dm': dm, 'epochs': epochs, 'window_size': window_size, 'vec_size': vec_size}
                    if accuracy2 > maxim:
                        maxim = accuracy2
                        max_params = { 'kernel': 'rbf', 'dm': dm, 'epochs': epochs, 'window_size': window_size, 'vec_size': vec_size}
                    print('-----------------------------------------')
    print('Max accuracy was', maxim, 'with params', max_params)

def get_cross_validated_baseline_accuracies(development_data, doc2vec_Xs, doc2vec_Y, doc2vec_models):
    ############# BOW accuracies ##############
    test_Y = development_data['sentiment'].values
    X, _ = get_bow_vectors(development_data['review'].values, min_count=4, max_frac=0.5)
    X_pres, _ = get_bow_vectors(development_data['review'].values, min_count=4, max_frac=0.5, frequency=False)
    X_low, _ = get_bow_vectors(development_data['review'].values, min_count=4, max_frac=0.5, lowercase=False, frequency=False)
    X_bi, _ = get_bow_vectors(development_data['review'].values, min_count=7, max_frac=0.5, frequency=False, bigrams=True)
    Y = development_data['sentiment'].to_numpy()
    acc6 = cross_validate_svm(X_bi, Y, kernel='linear', gamma='scale')
    print('Cross validated bow accuracy when using a linear kernel, feature presence and bigrams', acc6)
    acc7 = cross_validate_svm(np.concatenate((X_bi, X_pres), axis=1), Y, kernel='linear', gamma='scale')
    print('Cross validated bow accuracy when using a linear kernel, feature presence and both unigrams and bigrams', acc7)
    acc5 = cross_validate_svm(X_low, Y, kernel='linear', gamma='scale')
    print('Cross validated bow accuracy when using a linear kernel, feature presence and lowercased input', acc5)
    acc4 = cross_validate_svm(X_pres, Y, kernel='rbf', gamma='scale')
    print('Cross validated bow accuracy when using a gaussian kernel and feature presence', acc4)
    acc3 = cross_validate_svm(X_pres, Y, kernel='linear', gamma='scale')
    print('Cross validated bow accuracy when using a linear kernel and feature presence', acc3)
    acc1 = cross_validate_svm(X, Y, kernel='rbf', gamma='scale')
    print('Cross validated bow accuracy when using a gaussian kernel', acc1)
    acc2 = cross_validate_svm(X, Y, kernel='linear')
    print('Cross validated bow accuracy when using a linear kernel', acc2)
    ############# Doc2Vec accuracies #############
    svm3 = build_svm_classifier(doc2vec_Xs[2], doc2vec_Y, kernel='rbf', gamma='scale')
    test_X3 = get_doc2vec_data(development_data['review'].values, doc2vec_models[2])
    accuracy3 = estimate_svm_accuracy(test_X3, test_Y, svm3)
    print('Doc2Vec accuracy with a gaussian kernel and dm concat vectors', accuracy3)
    svm4 = build_svm_classifier(doc2vec_Xs[2], doc2vec_Y, kernel='linear', gamma='scale')
    test_X4 = get_doc2vec_data(development_data['review'].values, doc2vec_models[2])
    test_Y = development_data['sentiment'].values
    accuracy4 = estimate_svm_accuracy(test_X4, test_Y, svm4)
    print('Doc2Vec accuracy with a linear kernel and dm concat vectors', accuracy4)
    svm = build_svm_classifier(doc2vec_Xs[0], doc2vec_Y, kernel='rbf', gamma='scale')
    test_X = get_doc2vec_data(development_data['review'].values, doc2vec_models[0])
    accuracy = estimate_svm_accuracy(test_X, test_Y, svm)
    print('Doc2Vec accuracy with a gaussian kernel and dbow', accuracy)
    svm = build_svm_classifier(doc2vec_Xs[0], doc2vec_Y, kernel='linear', gamma='scale')
    test_X = get_doc2vec_data(development_data['review'].values, doc2vec_models[0])
    accuracy = estimate_svm_accuracy(test_X, test_Y, svm)
    print('Doc2Vec accuracy with a linear kernel and dbow', accuracy)
    test_X2 = get_doc2vec_data(development_data['review'].values, doc2vec_models[1])
    test_concat_X = np.concatenate((test_X, test_X2), axis=1)
    svm2 = build_svm_classifier(np.concatenate((doc2vec_Xs[0], doc2vec_Xs[1]), axis=1), doc2vec_Y, kernel='rbf', gamma='scale')
    accuracy2 = estimate_svm_accuracy(test_concat_X, test_Y, svm2)
    print('Doc2Vec accuracy with concatenated vectors and gaussian kernel', accuracy2)
    svm1 = build_svm_classifier(doc2vec_Xs[1], doc2vec_Y, kernel='linear', gamma='scale')
    test_X = get_doc2vec_data(development_data['review'].values, doc2vec_models[1])
    accuracy1 = estimate_svm_accuracy(test_X, test_Y, svm1)
    print('Doc2Vec accuracy with a linear kernel and dm', accuracy1)
    svm3 = build_svm_classifier(doc2vec_Xs[1], doc2vec_Y, kernel='rbf', gamma='scale')
    test_X = get_doc2vec_data(development_data['review'].values, doc2vec_models[1])
    accuracy3 = estimate_svm_accuracy(test_X, test_Y, svm3)
    print('Doc2Vec accuracy with a gaussian kernel and dm', accuracy3)
    test_X2 = get_doc2vec_data(development_data['review'].values, doc2vec_models[1])
    test_concat_X = np.concatenate((test_X, test_X2), axis=1)
    svm4 = build_svm_classifier(np.concatenate((doc2vec_Xs[0], doc2vec_Xs[1]), axis=1), doc2vec_Y, kernel='linear', gamma='scale')
    accuracy4 = estimate_svm_accuracy(test_concat_X, test_Y, svm4)
    print('Doc2Vec accuracy with concatenated vectors and linear kernel', accuracy4)

def run_permutation_test_two_bow_kernels(train, test, kernel1, kernel2):
    """ Run permutation test of difference between two kernels when using BOW vectors. """
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    svm1 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel1, gamma='scale')
    svm2 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel2, gamma='scale')
    return permutation_test(test_Y, svm1.predict(bow_test_X), svm2.predict(bow_test_X))

def run_permutation_test_bow_with_bigrams(train, test, kernel):
    """ Run permutation test of difference between using only bigrmas and only unigrams. """
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bi_train_X, bi_vectorizer = get_bow_vectors(train['review'].values, min_count=7, max_frac=0.5, frequency=False, bigrams=True)
    bi_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=bi_vectorizer)
    uni_train_X, uni_vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False)
    uni_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=uni_vectorizer)
    svm1 = build_svm_classifier(bi_train_X, train_Y, kernel=kernel, gamma='scale')
    svm2 = build_svm_classifier(uni_train_X, train_Y, kernel=kernel, gamma='scale')
    return permutation_test(test_Y, svm1.predict(bi_test_X), svm2.predict(uni_test_X))

def run_permutation_test_bow_with_uni_and_bigrams(train, test, kernel):
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bi_train_X, bi_vectorizer = get_bow_vectors(train['review'].values, min_count=7, max_frac=0.5, frequency=False, bigrams=True)
    bi_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=bi_vectorizer)
    uni_train_X, uni_vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False, bigrams=False)
    uni_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=uni_vectorizer)
    svm1 = build_svm_classifier(uni_train_X, train_Y, kernel=kernel, gamma='scale')
    svm2 = build_svm_classifier(np.concatenate((uni_train_X, bi_train_X), axis=1), train_Y, kernel=kernel, gamma='scale')
    return permutation_test(test_Y, svm1.predict(uni_test_X), svm2.predict(np.concatenate((uni_test_X, bi_test_X), axis=1)))

def run_permutation_test_bow_feature_presence_vs_frequency(train, test, kernel):
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    X_pres, pres_vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False)
    X_pres_test, _ = get_bow_vectors(test['review'].values, vectorizer=pres_vectorizer)
    X, vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5)
    X_test, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    svm1 = build_svm_classifier(X_pres, train_Y, kernel=kernel, gamma='scale')
    svm2 = build_svm_classifier(X, train_Y, kernel=kernel, gamma='scale')
    return permutation_test(test_Y, svm1.predict(X_pres_test), svm2.predict(X_test))

def run_permutation_test_bow_lowercase(train, test, kernel):
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    low_bow_train_X, low_vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, lowercase=False, frequency=False)
    low_bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=low_vectorizer)
    svm1 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel, gamma='scale')
    svm2 = build_svm_classifier(low_bow_train_X, train_Y, kernel=kernel, gamma='scale')
    return permutation_test(test_Y, svm1.predict(bow_test_X), svm2.predict(low_bow_test_X))

def run_permutation_test_bow_vs_doc2vec(train, test, bow_kernel, doc2vec_kernel, doc2vec_train_X, doc2vec_train_Y, doc2vec_model):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare data for both svms
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    doc2vec_test_X = get_doc2vec_data(test['review'].values, doc2vec_model)
    # build models and predict
    bow_svm = build_svm_classifier(bow_train_X, train_Y, kernel=bow_kernel, gamma='scale')
    doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel=doc2vec_kernel, gamma='scale')
    # delete big variables no longer used
    del bow_train_X;
    return permutation_test(test_Y, bow_svm.predict(bow_test_X), doc2vec_svm.predict(doc2vec_test_X))

def run_permutation_test_uni_bi_bow_vs_doc2vec(train, test, bow_kernel, doc2vec_kernel, doc2vec_train_X, doc2vec_train_Y, doc2vec_model):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare data for both svms
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bi_train_X, bi_vectorizer = get_bow_vectors(train['review'].values, min_count=7, max_frac=0.5, frequency=False, bigrams=True)
    bi_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=bi_vectorizer)
    uni_train_X, uni_vectorizer = get_bow_vectors(train['review'].values, min_count=4, max_frac=0.5, frequency=False, bigrams=True)
    uni_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=uni_vectorizer)
    doc2vec_test_X = get_doc2vec_data(test['review'].values, doc2vec_model)
    # build models and predict
    bow_svm = build_svm_classifier(np.concatenate((uni_train_X, bi_train_X), axis=1), train_Y, kernel=bow_kernel, gamma='scale')
    doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel=doc2vec_kernel, gamma='scale')
    # delete big variables no longer used
    return permutation_test(test_Y, bow_svm.predict(np.concatenate((uni_test_X, bi_test_X), axis=1)), doc2vec_svm.predict(doc2vec_test_X))

def run_permutation_test_two_different_doc2vecs(train, test, kernel1, kernel2, model1, model2, d2v_X_1, d2v_X_2, d2v_Y1, d2v_Y2):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare test data for both svms
    test_X1 = get_doc2vec_data(test['review'].values, model1)
    test_X2 = get_doc2vec_data(test['review'].values, model2)
    # build models and predict
    svm1 = build_svm_classifier(d2v_X_1, d2v_Y1, kernel=kernel1, gamma='scale')
    svm2 = build_svm_classifier(d2v_X_2, d2v_Y2, kernel=kernel2, gamma='scale')
    test_Y = test['sentiment'].to_numpy()
    return permutation_test(test_Y, svm1.predict(test_X1), svm2.predict(test_X2))

def run_permutation_test_two_doc2vec_kernels(train, test, kernel1, kernel2, model, d2v_X, d2v_Y, **kwargs):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare test data
    test_X = get_doc2vec_data(test['review'].values, model)
    # build models and predict
    svm1 = build_svm_classifier(d2v_X, d2v_Y, kernel=kernel1, gamma='scale', **kwargs)
    svm2 = build_svm_classifier(d2v_X, d2v_Y, kernel=kernel2, gamma='scale')
    test_Y = test['sentiment'].to_numpy()
    return permutation_test(test_Y, svm1.predict(test_X), svm2.predict(test_X))

def run_permutation_concatenated_vs_simple_doc2vec(train, test, kernel1, kernel2, model1, model2, d2v_X_1, d2v_X_2, d2v_Y):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare test data for both svms
    test_X1 = get_doc2vec_data(test['review'].values, model1)
    test_X2 = get_doc2vec_data(test['review'].values, model2)
    test_concat_X = np.concatenate((test_X1, test_X2), axis=1)
    # build models and predict
    svm1 = build_svm_classifier(d2v_X_1, d2v_Y, kernel=kernel1, gamma='scale')
    svm2 = build_svm_classifier(np.concatenate((d2v_X_1, d2v_X_2), axis=1), d2v_Y, kernel=kernel2, gamma='scale')
    test_Y = test['sentiment'].to_numpy()
    return permutation_test(test_Y, svm1.predict(test_X1), svm2.predict(test_concat_X))

def cross_validate_permutation_tests(training_data, imdb_reviews, doc2vec_Xs, doc2vec_Y, doc2vec_models):
    mean_p = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_feature_presence_vs_frequency, 
                                    kernel='rbf')
    print('mean p value with bow using unigrams and feature frequency vs presence', mean_p)
    print('Comparing different bow hyperparameters')
    mean_p2 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_with_uni_and_bigrams, 
                                    kernel='rbf')
    print('mean p value with bow using uni and bigrams and only unigrams and feature presence', mean_p2)
    mean_p3 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_two_bow_kernels, 
                                    kernel1='linear', 
                                    kernel2='rbf')
    print('mean p value with bow linear and rbf kernel when using feature presence', mean_p3)
    mean_p4 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_lowercase, 
                                    kernel='rbf')
    print('mean p value with rbf kernel and lowercase vs non-lowercased', mean_p4)
    #-----doc2vec stuff------
    mean_p5 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_vs_doc2vec, 
                                    bow_kernel='linear', 
                                    doc2vec_kernel='rbf', 
                                    doc2vec_train_X=doc2vec_Xs[0], 
                                    doc2vec_train_Y=doc2vec_Y, 
                                    doc2vec_model=doc2vec_models[0])
    print('mean p value with bow linear with presence and doc2vec rbf kernel',  mean_p5)
    print('Comparing doc2vec to bow with different kernels')
    mean_p6 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_two_doc2vec_kernels, 
                                    kernel1='linear', 
                                    kernel2='rbf', 
                                    d2v_X=doc2vec_Xs[0], 
                                    d2v_Y=doc2vec_Y, 
                                    model=doc2vec_models[0])
    print('mean p value with between doc2vec linear and gaussian kernels with dbow vectors',  mean_p6)
    mean_p12 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_two_doc2vec_kernels, 
                                    kernel1='linear', 
                                    kernel2='poly', 
                                    d2v_X=doc2vec_Xs[0], 
                                    d2v_Y=doc2vec_Y, 
                                    model=doc2vec_models[0],
                                    degree=3)
    print('mean p value with between doc2vec linear and polynomial kernels with dbow vectors',  mean_p12)
    mean_p7 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_uni_bi_bow_vs_doc2vec, 
                                    bow_kernel='linear', 
                                    doc2vec_kernel='rbf', 
                                    doc2vec_train_X=doc2vec_Xs[0], 
                                    doc2vec_train_Y=doc2vec_Y, 
                                    doc2vec_model=doc2vec_models[0])
    print('mean p value with bow linear with presence, uni and bigrams and doc2vec rbf kernel',  mean_p7)
    # compare dm to dbow 
    mean_p8 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_test_two_different_doc2vecs, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_models[0],
                                                model2=doc2vec_models[2],
                                                d2v_X_1=doc2vec_Xs[0],
                                                d2v_X_2=doc2vec_Xs[2],
                                                d2v_Y1=doc2vec_Y,
                                                d2v_Y2=doc2vec_Y)
    print('mean p value with dm concatenate and dbow and a gaussian kernel', mean_p8)
    mean_p9 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_test_two_different_doc2vecs, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_models[0], 
                                                model2=doc2vec_models[1], 
                                                d2v_X_1=doc2vec_Xs[0], 
                                                d2v_X_2=doc2vec_Xs[1], 
                                                d2v_Y1=doc2vec_Y,
                                                d2v_Y2=doc2vec_Y)
    print('mean p value with dm and dbow vectors', mean_p9)
    # compare vector sizes
    print('Comparing vector sizes in doc2vec')
    bigvec_doc2vec_train_X, bigvec_doc2vec_train_Y, bigvec_doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=25, vec_size=200, window_size=12, dm=0)
    mean_p10 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_test_two_different_doc2vecs, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_models[0],
                                                model2=bigvec_doc2vec_model,
                                                d2v_X_1=doc2vec_Xs[0],
                                                d2v_X_2=bigvec_doc2vec_train_X,
                                                d2v_Y1=doc2vec_Y,
                                                d2v_Y2=bigvec_doc2vec_train_Y)
    print('mean p value with bow vector sizes 100 and 200', mean_p10)
    # compare concatenation to just using dbow
    mean_p11 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_concatenated_vs_simple_doc2vec, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_models[0], 
                                                model2=doc2vec_models[1], 
                                                d2v_X_1=doc2vec_Xs[0], 
                                                d2v_X_2=doc2vec_Xs[1], 
                                                d2v_Y=doc2vec_Y)
    print('mean p value with dbow vs concatenated vectors', mean_p11)
    print('-------')

def sample_random_reviews_and_print(reviews, n=10):
    review_indices = np.random.choice(len(reviews), size=n, replace=False)
    revs = reviews['review'].values
    labels = reviews['sentiment'].values
    for i in review_indices:
        print(revs[i])
        print('label:', labels[i])
        print('index', i)
        print('---------------')

def analyse_emotion_statements(emotion_statement, statement, doc2vec_model):
    emotion_vector  = get_doc2vec_data([emotion_statement], doc2vec_model)
    vector  = get_doc2vec_data([statement], doc2vec_model)
    heat_plot_two_vectors(emotion_vector, emotion_statement, vector, statement)

def run_intensification_analysis(doc2vec_model):
    emotion_statement = 'this is the best movie ever made .'
    statement = 'this is the worst movie ever made .'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)
    emotion_statement = 'this is the best movie ever made .'
    statement = 'this is is a movie .'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)
    emotion_statement = 'i like it .'
    statement = 'i like it a lot .'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)
    emotion_statement = 'i hate this movie .'
    statement = 'i hate this movie so much .'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)
    emotion_statement = 'not good .'
    statement = 'good .'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)
    emotion_statement = 'like'
    statement = 'n\'t like'
    analyse_emotion_statements(emotion_statement, statement, doc2vec_model)

def do_test_visualisations(blind_test_set, conc_test_X, test_Y, doc2vec_test_X, doc2vec_model):
    vis_indices = [28, 164, 37, 100, 186, 41, 64, 17, 54, 80]
    for i in vis_indices:
        print('i:', i)
        print(blind_test_set['review'].iloc[i])
        print('------------')
    bow_distance_matrix = calculate_vector_distances(conc_test_X[vis_indices], conc_test_X[vis_indices])
    descriptions = ['negative (contradictory)']*3 + ['really negative']*4 + ['praising']*3
    short_descriptions = ['neg (contr)']*3 + ['neg']*4 + ['pos']*3
    plot_matrix(bow_distance_matrix, axislabels=short_descriptions, title='Bow distance matrix')
    doc2vec_distance_matrix = calculate_vector_distances(doc2vec_test_X[vis_indices], doc2vec_test_X[vis_indices])
    plot_matrix(doc2vec_distance_matrix, axislabels=short_descriptions, title='Doc2Vec distance matrix')
    texts = [str(i) + ': ' + d for i, d in zip(vis_indices, descriptions)]
    visualise_individual_reviews(vis_indices, imdb_reviews, doc2vec_model)
    visualize_vectors(conc_test_X[vis_indices], test_Y[vis_indices], texts, use_pca=False, perplexity=10, n_iter=1000, title='Hand picked reviews plotted together (BOW)')
    visualize_vectors(doc2vec_test_X[vis_indices], test_Y[vis_indices], texts, use_pca=False, pca_components=5, perplexity=10, n_iter=1000, title='Hand picked reviews plotted together (Doc2Vec)')
    visualize_vectors(conc_test_X, test_Y, title='BOW blind test set vectors')
    visualize_vectors(doc2vec_test_X, test_Y, title='Doc2Vec blind test set vectors')

def deployment_test(imdb_data_folder, doc2vec_model, doc2vec_svm, uni_vectorizer, bi_vectorizer, uni_bi_bow_svm):
    print('Fetching new reviews')
    imdb_data_folder = 'aclImdb'
    imdb_sentiments = ['pos', 'neg']
    # get new imdb reviews
    new_reviews = get_reviews(imdb_data_folder, imdb_sentiments, ['new'])
    new_test_Y = new_reviews['sentiment']
    doc2vec_test_new = get_doc2vec_data(new_reviews['review'].values, doc2vec_model)
    doc2vec_acc_new = estimate_svm_accuracy(doc2vec_test_new, new_test_Y, doc2vec_svm)
    print('Doc2Vec acc with new data', doc2vec_acc_new)
    new_test_X, _ = get_bow_vectors(new_reviews['review'].values, min_count=4, max_frac=0.5, frequency=False, vectorizer=uni_vectorizer)
    new_bi_test_X, _ = get_bow_vectors(new_reviews['review'].values, min_count=7, max_frac=0.5, frequency=False, vectorizer=bi_vectorizer)
    new_conc_test_X = np.concatenate((new_test_X, new_bi_test_X), axis=1)
    bow_acc_new = estimate_svm_accuracy(new_conc_test_X, new_test_Y, uni_bi_bow_svm)
    print('BOW acc with new data', bow_acc_new)
    for i in range(len(new_reviews)):
        print('i:', i)
        print('review')
        print(new_reviews['review'].iloc[i])
        print('bow prediction', uni_bi_bow_svm.predict(new_conc_test_X[i].reshape(1, -1)))
        print('doc2vec prediction', doc2vec_svm.predict(doc2vec_test_new[i].reshape(1, -1)))
        print('correct label', new_reviews['sentiment'].iloc[i])
        print('-------------')
def get_bow_data(review_data, train_frac=0.7, min_count=10, max_frac=0.5, dim=100):
    """ Get BOW vector training and test sets. """
    X = get_bow_vectors(review_data['review'].values, min_count, max_frac)
    Y = review_data['sentiment'].to_numpy()
    print('Created a BOW vector of shape', X.shape)
    return get_train_test_split(0.7, X, Y)
    


def main():
    np.random.seed(123)
    imdb_data_folder = 'aclImdb'
    imdb_sentiments = ['pos', 'neg']
    subfolders = ['train', 'test']
    # get imdb reviews to train doc2vec with
    imdb_reviews = get_reviews(imdb_data_folder, imdb_sentiments, subfolders)
    reviews, _ = get_uni_and_bi_grams('data-tagged')
    review_data = build_data(reviews)
    # set a blind set aside for reporting results
    development_data, blind_test_set = get_train_test_split(0.9, review_data)
    test_Y = blind_test_set['sentiment'].values
    #find_optimal_doc2vec_hyperparams(imdb_reviews, development_data)
    ####### TRAINING MODELS AND RUNNING EXPERIMENTS #############
    doc2vec_train_X, doc2vec_train_Y, doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=30, window_size=4, dm=0, dbow_words=1, pretrained=True, save=True)
    doc2vec_test_X = get_doc2vec_data(blind_test_set['review'].values, doc2vec_model)
    doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel='rbf', gamma='scale')
    concat_doc2vec_train_X, concat_doc2vec_train_Y, concat_doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=30, window_size=4, dm=1, dm_concat=1, pretrained=True, save=True)
    dm_doc2vec_train_X, dm_doc2vec_train_Y, dm_doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=30, window_size=4, dm=1, pretrained=True, save=True)
    print('-----------')
    doc2vec_train_Xs = [doc2vec_train_X, dm_doc2vec_train_X, concat_doc2vec_train_X]
    doc2vec_models = [doc2vec_model, dm_doc2vec_model, concat_doc2vec_model]
    get_cross_validated_baseline_accuracies(development_data, 
                                            doc2vec_Xs=doc2vec_train_Xs,
                                            doc2vec_Y=doc2vec_train_Y,
                                            doc2vec_models=doc2vec_models)
    print('-----------')
    cross_validate_permutation_tests(development_data, imdb_reviews, doc2vec_train_Xs, doc2vec_train_Y, doc2vec_models)
    ###################### USING THE BEST MODELS #################
    print('---------Accuracies using the best models---------')
    doc2vec_acc = estimate_svm_accuracy(doc2vec_test_X, test_Y, doc2vec_svm)
    print('doc2vec accuracy', doc2vec_acc)
    X_pres, uni_vectorizer = get_bow_vectors(development_data['review'].values, min_count=4, max_frac=0.5, frequency=False)
    X_bi, bi_vectorizer = get_bow_vectors(development_data['review'].values, min_count=7, max_frac=0.5, frequency=False, bigrams=True)
    test_X, _ = get_bow_vectors(blind_test_set['review'].values, min_count=4, max_frac=0.5, frequency=False, vectorizer=uni_vectorizer)
    bi_test_X, _ = get_bow_vectors(blind_test_set['review'].values, min_count=7, max_frac=0.5, frequency=False, vectorizer=bi_vectorizer)
    conc_test_X = np.concatenate((test_X, bi_test_X), axis=1)
    uni_bi_bow_svm = build_svm_classifier(np.concatenate((X_pres, X_bi), axis=1), development_data['sentiment'].values, kernel='linear', probability=True)
    bow_acc = estimate_svm_accuracy(conc_test_X, test_Y, uni_bi_bow_svm)
    print('bow accuracy with presence and both uni and bigrams and a linear kernel', bow_acc)
    bow_svm = build_svm_classifier(X_pres, development_data['sentiment'].values, kernel='linear')
    bow_acc2 = estimate_svm_accuracy(test_X, test_Y, bow_svm)
    print('bow accuracy with presence, unigrams and a linear kernel', bow_acc2)
    print('-------------')
    print('Error analysis for Doc2Vec')
    model_error_analysis(doc2vec_test_X, blind_test_set, doc2vec_svm)
    evaluate_vector_qualities(doc2vec_test_X, test_Y)
    print('Error analysis for BOW with both unigrams and bigrams')
    model_error_analysis(conc_test_X, blind_test_set, uni_bi_bow_svm)
    evaluate_vector_qualities(conc_test_X, test_Y)
    print('----------------')
    print('Vector quality estimation')
    evaluate_vector_qualities(test_X, test_Y, model_imdb)
    do_test_visualisations(blind_test_set, conc_test_X, test_Y, doc2vec_test_X, doc2vec_model)
    print('----------------')
    print('Plotting emotion sentences for intensification analysis')
    run_intensification_analysis(doc2vec_model)
    print('Running a deployment test')
    deployment_test(imdb_data_folder, doc2vec_model, doc2vec_svm, uni_vectorizer, bi_vectorizer, uni_bi_bow_svm)

if __name__ == '__main__':
    main()