import re
import gensim
from gensim.test.utils import common_texts
import os
import pandas as pd
from sklearn import svm
from joblib import dump, load
from doc_utils import doc_tokenize, get_reviews, train_doc2vec_model, get_doc2vec_data, get_bow_vectors, visualize_vectors, evaluate_vector_qualities
from science_utils import sample_variance, get_mask, get_train_test_split, train_test_split_indexes, get_accuracy, permutation_test
from ngram_utils import get_uni_and_bi_grams
import time
import numpy as np
from practical1 import build_data
from sklearn.manifold import TSNE
from scipy import spatial
from visualisation import plot_heat_map
import itertools

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
        indices = rand_perm[f:f*step+step] if f*step+step < len(rand_perm) else rand_perm[-step:]
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
        indices = rand_perm[f:f*step+step] if f*step+step < len(rand_perm) else rand_perm[-step:]
        test_mask = get_mask(len(data), indices)
        test = data[test_mask]
        train = data[~test_mask]
        p_value = run_perm_test(train, test, **kwargs)
        print(p_value)
        p_values[f] = p_value
    # get p-value for all folds
    return np.mean(p_values), sample_variance(p_values)

def run_permutation_test_on_one_kind_of_data(test_X, test_Y, svm1, smv2):
    preds1 = svm1.predict(test_X)
    preds2 = smv2.predict(test_X)
    return permutation_test(test_Y, preds1, preds2)

def run_permutation_test_bow_vs_doc2vec(train, test, bow_kernel, doc2vec_kernel, bow_C, doc2vec_C, doc2vec_train_X, doc2vec_train_Y, doc2vec_model):
    """ This is used to compare two svms which use a different transform on the dataset, e.g. bow versus doc2vec. """
    # prepare training data for both svms
    train_Y = train['sentiment'].to_numpy()
    test_Y = test['sentiment'].to_numpy()
    bow_train_X, vectorizer = get_bow_vectors(train['review'].values, min_count=5, max_frac=0.5)
    bow_test_X, _ = get_bow_vectors(test['review'].values, vectorizer=vectorizer)
    doc2vec_test_X = get_doc2vec_data(test['review'].values, doc2vec_model)
    # build models and predict
    bow_svm = build_svm_classifier(bow_train_X, train_Y, kernel=bow_kernel, C=bow_C, gamma='scale')
    doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel=doc2vec_kernel, C=doc2vec_C, gamma='scale')
    return permutation_test(test_Y, bow_svm.predict(bow_test_X), doc2vec_svm.predict(doc2vec_test_X))

def doc2vec_error_analysis(test_data, svm, doc2vec_model):
    """ Print out misclassified instances with their prediction and true label. """
    test_X = get_doc2vec_data(test_data['review'].values, doc2vec_model)    
    preds = svm.predict(test_X)
    false_mask = (preds != test_data['sentiment'].values)
    false_indices = np.where(false_mask)[0]
    # choose 5 random indices and print
    indices = false_indices[np.random.choice(len(false_indices), size=5, replace=False)]
    for i in indices:
        print('Review')
        print(test_data['review'].iloc[i])
        print('True label', test_data['sentiment'].iloc[i])
        print('Classified as', preds[i])
        print('-----------------')
    # compare the vectors of the falsely classified instances to support vectors
    # use svm.support_vectors_ to get the vectors
    support_vectors = svm.support_vectors_
    false_distances = [spatial.distance.cosine(supp, vec) for vec in test_X[false_mask] for supp in support_vectors]
    true_distances = [spatial.distance.cosine(supp, vec) for vec in test_X[~false_mask] for supp in support_vectors]
    sv_dist_from_misclassified = np.mean(false_distances), np.var(false_distances)
    sv_dist_from_others = np.mean(true_distances), np.var(true_distances)
    print('sv_dist_from_misclassified', sv_dist_from_misclassified)
    print('sv_dist_from_others', sv_dist_from_others)

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
    for window_size in [4, 8, 10, 12]:
        for epochs in [25, 30, 35]:
            for dm in [0, 1]:
                for vec_size in [50, 100, 150, 200]:
                    print('-----------------------------------------')
                    print('window size', window_size, 'epochs', epochs, 'dm', dm, 'vec size', vec_size)
                    train_X, train_Y, model_imdb = train_doc2vec_model(imdb_reviews, epochs=epochs, vec_size=vec_size, window_size=window_size, dm=dm)
                    #visualize_vectors(train_X, train_Y, window_size, epochs)
                    test_X = get_doc2vec_data(dev_data['review'].values, model_imdb)
                    test_Y = dev_data['sentiment'].values
                    #visualize_vectors(test_X, test_Y, window_size, epochs)
                    print('For test set vectors inferred with doc2vec')
                    evaluate_vector_qualities(test_X, test_Y, model_imdb)
                    svm1 = build_svm_classifier(train_X, train_Y, kernel='linear')
                    accuracy1 = estimate_svm_accuracy(test_X, test_Y, svm1)
                    print('accuracy using doc2vec and svm with a linear kernel', accuracy1)
                    svm2 = build_svm_classifier(train_X, train_Y, kernel='poly', gamma='scale')
                    accuracy2 = estimate_svm_accuracy(test_X, test_Y, svm2)
                    print('accuracy using doc2vec and svm with a poly (4) kernel', accuracy2)
                    svm3 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale')
                    accuracy3 = estimate_svm_accuracy(test_X, test_Y, svm3)
                    print('accuracy using doc2vec and svm with a gaussian kernel', accuracy3)
                    if accuracy1 > maxim:
                        maxim = accuracy1
                        max_params = { 'kernel': 'linear', 'dm': dm, 'epochs': epochs, 'window_size': window_size}
                    if accuracy2 > maxim:
                        maxim = accuracy2
                        max_params = { 'kernel': 'poly', 'dm': dm, 'epochs': epochs, 'window_size': window_size}
                    if accuracy3 > maxim:
                        maxim = accuracy3
                        max_params = { 'kernel': 'rbf', 'dm': dm, 'epochs': epochs, 'window_size': window_size}
                    print('-----------------------------------------')
    print('Max accuracy was', maxim, 'with params', max_params)


def main():
    imdb_data_folder = 'aclImdb'
    imdb_sentiments = ['pos', 'neg']
    subfolders = ['train', 'test']
    reviews, _ = get_uni_and_bi_grams('data-tagged')
    review_data = build_data(reviews)
    # set a blind set aside for reporting results
    development_data, blind_test = get_train_test_split(0.9, review_data)
    imdb_reviews = get_reviews(imdb_data_folder, imdb_sentiments, subfolders)
    doc2vec_train_X, doc2vec_train_Y, doc2vec_model = train_doc2vec_model(imdb_reviews, epochs=25, window_size=4, dm=0)
    training_data, val_data = get_train_test_split(0.7, development_data)
    print('Training data size', len(training_data), 'test data size', len(val_data))
    X = get_bow_vectors(development_data['review'].values, min_count=5, max_frac=0.5)
    Y = development_data['sentiment'].to_numpy()
    '''
    acc1 = cross_validate_svm(X, Y, kernel='rbf', gamma='scale', C=4.7)
    print('Cross validated accuracy when using a gaussian kernel', acc1)
    acc2 = cross_validate_svm(X, Y, kernel='linear', C=8.3)
    print('Cross validated accuracy when using a linear kernel', acc2)
    svm1 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale')
    svm2 = build_svm_classifier(train_X, train_Y, kernel='linear', gamma='scale')  
    perm_p = run_permutation_test_bow_vs_doc2vec(training_data, 
                                                val_data, 
                                                bow_kernel='linear', 
                                                doc2vec_kernel='linear', 
                                                bow_C=4.7, doc2vec_C=4.7, 
                                                doc2vec_train_X=doc2vec_train_X, 
                                                doc2vec_train_Y=doc2vec_train_Y, 
                                                doc2vec_model=doc2vec_model)
    print('bow linear, doc2vec linear', perm_p)
    perm_p = run_permutation_test_bow_vs_doc2vec(training_data, 
                                                val_data, 
                                                bow_kernel='linear', 
                                                doc2vec_kernel='rbf', 
                                                bow_C=4.7, doc2vec_C=4.7, 
                                                doc2vec_train_X=doc2vec_train_X, 
                                                doc2vec_train_Y=doc2vec_train_Y, 
                                                doc2vec_model=doc2vec_model)
    print('bow linear, doc2vec rbf', perm_p)
    #doc2vec_svm = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel='rbf', gamma='scale')
    #doc2vec_error_analysis(val_data, doc2vec_svm, doc2vec_model)    
    mean_p2 = cross_validate_permutation_test(training_data,
                                                10, 
                                                run_permutation_test_bow_vs_doc2vec, 
                                                bow_kernel='linear', 
                                                doc2vec_kernel='linear', 
                                                bow_C=4.7, doc2vec_C=4.7, 
                                                doc2vec_train_X=doc2vec_train_X, 
                                                doc2vec_train_Y=doc2vec_train_Y, 
                                                doc2vec_model=doc2vec_model)
    print('mean p value with bow linear kernel and doc2vec linear kernel', mean_p2)
    mean_p3 = cross_validate_permutation_test(training_data,
                                    10, 
                                    run_permutation_test_bow_vs_doc2vec, 
                                    bow_kernel='linear', 
                                    doc2vec_kernel='rbf', 
                                    bow_C=4.7, doc2vec_C=4.7, 
                                    doc2vec_train_X=doc2vec_train_X, 
                                    doc2vec_train_Y=doc2vec_train_Y, 
                                    doc2vec_model=doc2vec_model)
    print('mean p value with bow linear and doc2vec rbf kernel',  mean_p3)
    acc2 = cross_validate_svm(X, Y, kernel='linear', C=8.3)
    print('Cross validated accuracy when using bow vectors and a linear kernel', acc2)
    svm3 = build_svm_classifier(doc2vec_train_X, doc2vec_train_Y, kernel='rbf', gamma='scale', C=4.7)
    test_X = get_doc2vec_data(val_data['review'].values, doc2vec_model)
    test_Y = val_data['sentiment'].values
    accuracy3 = estimate_svm_accuracy(test_X, test_Y, svm3)
    print('doc2vec accuracy', accuracy3)
    '''
    find_optimal_doc2vec_hyperparams(imdb_reviews, val_data)


if __name__ == '__main__':
    main()