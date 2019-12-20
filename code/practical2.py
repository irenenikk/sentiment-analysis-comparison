import re
import gensim
from gensim.test.utils import common_texts
import os
import pandas as pd
from sklearn import svm
from joblib import dump, load
from doc_utils import doc_tokenize, get_reviews, train_doc2vec_model, get_doc2vec_data, get_bow_vectors, visualize_vectors, evaluate_vector_qualities, doc2vec_error_analysis
from science_utils import sample_variance, get_mask, get_train_test_split, train_test_split_indexes, get_accuracy, permutation_test
from ngram_utils import get_uni_and_bi_grams
import time
import numpy as np
from practical1 import build_data
from sklearn.manifold import TSNE
from scipy import spatial
from visualisation import plot_heat_map
import itertools
import sys

models_file = 'models'

def build_svm_classifier(X, Y, pretrained=False, save=False, **kwargs):
    idd = '_'.join(['{}={}'.format(key, str(value)) for key, value in kwargs.items()])
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

def run_permutation_test_two_bow_kernels(train, test, kernel1, kernel2):
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=5, max_frac=0.5, frequency=False)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    svm1 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel1, gamma='scale')
    svm2 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel2, gamma='scale')
    return permutation_test(test_Y, svm2.predict(bow_test_X), svm2.predict(bow_test_X))

def run_permutation_test_bow_lowercase(train, test, kernel):
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=5, max_frac=0.5)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    low_bow_train_X, low_vectorizer = get_bow_vectors(train['review'].values, min_count=5, max_frac=0.5, lowercase=False, frequency=False)
    low_bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=low_vectorizer)
    svm1 = build_svm_classifier(bow_train_X, train_Y, kernel=kernel, gamma='scale')
    svm2 = build_svm_classifier(low_bow_train_X, train_Y, kernel=kernel, gamma='scale')
    return permutation_test(test_Y, svm2.predict(bow_test_X), svm2.predict(low_bow_test_X))

def run_permutation_test_bow_vs_doc2vec(train, test, bow_kernel, doc2vec_kernel, bow_C, doc2vec_train_X, doc2vec_train_Y, doc2vec_model):
    """ Run a permutation test to compare bow and doc2vec svms. """
    # prepare data for both svms
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=5, max_frac=0.5, frequency=False)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    doc2vec_test_X = get_doc2vec_data(test['review'].values, doc2vec_model)
    # build models and predict
    bow_svm = build_svm_classifier(bow_train_X, train_Y, kernel=bow_kernel, C=bow_C, gamma='scale')
    doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel=doc2vec_kernel, gamma='scale')
    # delete big variables no longer used
    del bow_train_X;
    return permutation_test(test_Y, bow_svm.predict(bow_test_X), doc2vec_svm.predict(doc2vec_test_X))

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

def find_optimal_rbf_kernel_parameters(X, Y, c_range, kernel):
    """ Find optimal C and Gamma for a Gaussian kernel and visualize results. """
    c_range = np.arange(0.1, 10, 0.2)
    maxx = -1
    best_c = -1
    for c in c_range:
            accuracy = cross_validate_svm(X, Y, kernel=kernel, C=c)
            if accuracy[0] > maxx:
                maxx = accuracy[0]
                best_c = c
    print('Best accuracy with a linear kernel', maxx, 'and gamma = scale, and c =', best_c)
    return best_c

def find_optimal_doc2vec_hyperparams(imdb_reviews, dev_data):
    maxim = -1
    max_params = {}
    for window_size in [4, 10, 12]:
        for epochs in [25, 35]:
            for dm in [0]:
                for vec_size in [200]:
                    print('-----------------------------------------')
                    print('window size', window_size, 'epochs', epochs, 'dm', dm, 'vec size', vec_size)
                    train_X, train_Y, model_imdb = train_doc2vec_model(imdb_reviews, epochs=epochs, vec_size=vec_size, window_size=window_size, dm=dm)
                    #visualize_vectors(train_X, train_Y, window_size, epochs)
                    test_X = get_doc2vec_data(dev_data['review'].values, model_imdb)
                    test_Y = dev_data['sentiment'].values
                    #visualize_vectors(test_X, test_Y, window_size, epochs)
                    print('For test set vectors inferred with doc2vec')
                    #evaluate_vector_qualities(test_X, test_Y, model_imdb)
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

def get_cross_validated_baseline_accuracies(development_data, d2v_X_1, d2v_X_2, d2v_Y, d2v_model1, d2v_model2):
    # bow baseline
    X, _ = get_bow_vectors(development_data['review'].values, min_count=5, max_frac=0.5)
    X_pres, _ = get_bow_vectors(development_data['review'].values, min_count=5, max_frac=0.5, frequency=False)
    Y = development_data['sentiment'].to_numpy()
    acc3 = cross_validate_svm(X_pres, Y, kernel='linear', gamma='scale')
    print('Cross validated bow accuracy when using a linear kernel and feature presence', acc3)
    acc4 = cross_validate_svm(X_pres, Y, kernel='rbf', gamma='scale')
    print('Cross validated bow accuracy when using a gaussian kernel and feature presence', acc4)
    acc1 = cross_validate_svm(X, Y, kernel='rbf', gamma='scale')
    print('Cross validated bow accuracy when using a gaussian kernel', acc1)
    acc2 = cross_validate_svm(X, Y, kernel='linear')
    print('Cross validated bow accuracy when using a linear kernel', acc2)
    # get doc2vec baseline accuracy
    svm = build_svm_classifier(d2v_X_1, d2v_Y, kernel='rbf', gamma='scale')
    test_X = get_doc2vec_data(development_data['review'].values, d2v_model1)
    test_Y = development_data['sentiment'].values
    accuracy3 = estimate_svm_accuracy(test_X, test_Y, svm)
    print('Doc2Vec accuracy with a gaussian kernel', accuracy3)
    test_X2 = get_doc2vec_data(development_data['review'].values, d2v_model2)
    test_concat_X = np.concatenate((test_X, test_X2), axis=1)
    svm2 = build_svm_classifier(np.concatenate((d2v_X_1, d2v_X_2), axis=1), d2v_Y, kernel='rbf', gamma='scale')
    accuracy2 = estimate_svm_accuracy(test_concat_X, test_Y, svm2)
    print('Doc2Vec accuracy with concatenated vectors', accuracy2)


def run_single_split_permutation_tests(training_data, val_data, doc2vec_train_X, doc2vec_train_Y, doc2vec_model):
    perm_p = run_permutation_test_bow_vs_doc2vec(training_data, 
                                                val_data, 
                                                bow_kernel='linear', 
                                                doc2vec_kernel='rbf', 
                                                bow_C=4.7, 
                                                doc2vec_train_X=doc2vec_train_X, 
                                                doc2vec_train_Y=doc2vec_train_Y, 
                                                doc2vec_model=doc2vec_model)
    print('bow linear, doc2vec rbf permutation test p-value', perm_p)

def cross_validate_permutation_tests(training_data, imdb_reviews, doc2vec_train_X, doc2vec_train_Y, doc2vec_model, dm_doc2vec_train_X, dm_doc2vec_train_Y, dm_doc2vec_model):
    print('Comparing different bow hyperparameters')
    mean_p = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_two_bow_kernels, 
                                    kernel1='linear', 
                                    kernel2='rbf')
    print('mean p value with bow linear and rbf kernel', mean_p)
    mean_p2 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_lowercase, 
                                    kernel='rbf')
    print('mean p value with bow linear and rbf kernel', mean_p2)
    print('Comparing doc2vec to bow with different kernels')
    mean_p3 = cross_validate_permutation_test(training_data,
                                    3, 
                                    run_permutation_test_bow_vs_doc2vec, 
                                    bow_kernel='linear', 
                                    doc2vec_kernel='rbf', 
                                    bow_C=4.7, 
                                    doc2vec_train_X=doc2vec_train_X, 
                                    doc2vec_train_Y=doc2vec_train_Y, 
                                    doc2vec_model=doc2vec_model)
    print('mean p value with bow linear and doc2vec rbf kernel',  mean_p3)
    # compare concatenation to just using dbow
    mean_p3 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_concatenated_vs_simple_doc2vec, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_model, 
                                                model2=dm_doc2vec_model, 
                                                d2v_X_1=doc2vec_train_X, 
                                                d2v_X_2=dm_doc2vec_train_X, 
                                                d2v_Y=doc2vec_train_Y)
    print('mean p value with dbow vs concatenated vectors', mean_p3)
    # compare dm to dbow 
    mean_p4 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_test_two_different_doc2vecs, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_model, 
                                                model2=dm_doc2vec_model, 
                                                d2v_X_1=doc2vec_train_X, 
                                                d2v_X_2=dm_doc2vec_train_X, 
                                                d2v_Y1=doc2vec_train_Y,
                                                d2v_Y2=dm_doc2vec_train_Y)
    print('mean p value with dm and dbow vectors', mean_p4)
    # compare vector sizes
    print('Comparing vector sizes in doc2vec')
    bigvec_doc2vec_train_X, bigvec_doc2vec_train_Y, bigvec_doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=25, vec_size=200, window_size=12, dm=0)
    mean_p5 = cross_validate_permutation_test(training_data,
                                                3, 
                                                run_permutation_test_two_different_doc2vecs, 
                                                kernel1='rbf', 
                                                kernel2='rbf',
                                                model1=doc2vec_model,
                                                model2=bigvec_doc2vec_model,
                                                d2v_X_1=doc2vec_train_X,
                                                d2v_X_2=bigvec_doc2vec_train_X,
                                                d2v_Y1=doc2vec_train_Y,
                                                d2v_Y2=bigvec_doc2vec_train_Y)
    print('mean p value with bow vector sizes 100 and 200', mean_p5)
    print('-------')

def get_blind_test_results(blind_test_set, doc2vec_model, doc2vec_svm, bow_svm, **bow_args):
    test_Y = blind_test_set['sentiment'].values
    test_X = get_doc2vec_data(blind_test_set['review'].values, doc2vec_model)
    doc2vec_acc = estimate_svm_accuracy(test_X, test_Y, doc2vec_svm)
    print('doc2vec accuracy', doc2vec_acc)
    bow_test_X, = get_bow_vectors(blind_test_set['review'].values, vectorizer, **bow_args)
    bow_acc = estimate_svm_accuracy(bow_test_X, test_Y, doc2vec_acc)
    print('bow accuracy', bow_acc)

def main():
    np.random.seed(42)
    imdb_data_folder = 'aclImdb'
    imdb_sentiments = ['pos', 'neg']
    subfolders = ['train', 'test']
    reviews, _ = get_uni_and_bi_grams('data-tagged')
    review_data = build_data(reviews)
    # set a blind set aside for reporting results
    development_data, blind_test = get_train_test_split(0.9, review_data)
    # get imdb reviews to train doc2vec with
    imdb_reviews = get_reviews(imdb_data_folder, imdb_sentiments, subfolders)
    # train baseline doc2vec model
    doc2vec_train_X, doc2vec_train_Y, doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=25, window_size=4, dm=0)
    dm_doc2vec_train_X, dm_doc2vec_train_Y, dm_doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=25, window_size=4, dm=1)
    # split development data into training and validation sets
    training_data, val_data = get_train_test_split(0.7, development_data)
    print('Training data size', len(training_data), 'test data size', len(val_data))
    print('-----------')
    #get_cross_validated_baseline_accuracies(development_data, d2v_X_1=doc2vec_train_X, d2v_X_2=dm_doc2vec_train_X, d2v_Y=doc2vec_train_Y, d2v_model1=doc2vec_model, d2v_model2=dm_doc2vec_model)
    print('-----------')
    #run_single_split_permutation_tests(training_data, val_data, doc2vec_train_X, doc2vec_train_Y, doc2vec_model)
    print('-----------')
    cross_validate_permutation_tests(training_data, imdb_reviews, doc2vec_train_X, doc2vec_train_Y, doc2vec_model, dm_doc2vec_train_X, dm_doc2vec_train_Y, dm_doc2vec_model)
    print('----------')
    # analyse doc2vec
    #doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel='rbf', gamma='scale')
    #doc2vec_error_analysis(val_data, doc2vec_svm, doc2vec_model)




if __name__ == '__main__':
    main()