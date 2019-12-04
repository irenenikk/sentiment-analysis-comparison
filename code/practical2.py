import re
import gensim
from gensim.test.utils import common_texts
import os
import pandas as pd
from sklearn import svm
from joblib import dump, load
from doc_utils import doc_tokenize, get_reviews, train_doc2vec_model, get_doc2vec_data, get_bow_vectors
from science_utils import sample_variance, get_mask, get_train_test_split, train_test_split_indexes, get_accuracy, permutation_test, permutation_test2
from ngram_utils import get_uni_and_bi_grams
import time
import numpy as np
from practical1 import build_data
from sklearn.manifold import TSNE
from scipy import spatial
from visualisation import plot_heat_map

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

def run_permutation_test_on_two_svm_kernels(test_X, test_Y, svm1, smv2):
    preds1 = svm1.predict(test_X)
    preds2 = smv2.predict(test_X)
    return permutation_test2(test_Y, preds1, preds2)

def gaussian_vs_linear_permutation_test(X, Y):
    gaussian_scores = cross_validate_svm(X, Y, return_raw=True, kernel='rbf', gamma='scale', C=4.7)
    print(gaussian_scores)
    linear_scores = cross_validate_svm(X, Y, return_raw=True, kernel='linear', gamma='scale', C=8.3)
    print(linear_scores)
    return permutation_test(gaussian_scores, linear_scores)

def run_permutation_test_on_two_svms(test_X, test_Y, transform1, transform2, svm1, smv2):
    """ This is used to compare two svms which use a different transform on the dataset, e.g. bow versus doc2vec. """
    preds1 = svm1.predict(transform1(test_X))
    preds2 = smv2.predict(transform2(test_X))
    return permutation_test(test_Y, preds1, preds2)

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

def get_bow_data(review_data, min_count=10, max_frac=0.5, dim=100):
    """ Get BOW vector training and test sets. """
    X = get_bow_vectors(review_data['review'].values, min_count, max_frac)
    Y = review_data['sentiment'].to_numpy()
    print('Created a BOW vector of shape', X.shape)
    return X, Y

def visualize_vectors(X, Y, window_size, epochs):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig = go.Figure(data=go.Scatter(x=X_embedded[:,0],
                                    y=X_embedded[:,1],
                                    mode='markers',
                                    marker_color=Y))
    fig.update_layout(title='Doc2Vec t-SNE representation, window size {}, epochs {}'.format(window_size, epochs))
    fig.show()

def evaluate_vector_qualities(X, Y, model):
    # 0: same label, 1: different label
    distances = np.zeros((len(X), 2))
    for i in range(len(X)):
        distances_to_others = np.zeros((len(X), 2))
        for j in range(len(X)):
            if i == j:
                continue
            distance = spatial.distance.cosine(X[i], X[j])
            if Y[i] == Y[j]:
                distances_to_others[j, 0] = distance
            else:
                distances_to_others[j, 1] = distance
        distances[i, 0] = np.mean(distances_to_others[:, 0])
        distances[i, 1] = np.mean(distances_to_others[:, 1])
    print('Average distance to all other vectors', np.mean(distances[:,1]))
    print('Average distance between reviews of same label', np.mean(distances[:,0]))

def find_optimal_doc2vec_hyperparams(imdb_reviews, dev_data):
    maxim = -1
    max_params = {}
    for window_size in [2, 4, 6, 8]:
        for epochs in [20, 25, 30, 35]:
            for dm in [0, 1]:
                print('-----------------------------------------')
                print('window size', window_size, 'epochs', epochs, 'dm', dm)
                train_X, train_Y, model_imdb = train_doc2vec_model(imdb_reviews, epochs=epochs, window_size=window_size, dm=dm)
                #visualize_vectors(train_X, train_Y, window_size, epochs)
                test_X = get_doc2vec_data(dev_data['review'].values, model_imdb)
                test_Y = dev_data['sentiment'].values
                visualize_vectors(test_X, test_Y, window_size, epochs)
                print('For test set vectors inferred with doc2vec')
                evaluate_vector_qualities(test_X, test_Y, model_imdb)
                svm1 = build_svm_classifier(train_X, train_Y, kernel='linear')
                accuracy1 = estimate_svm_accuracy(test_X, test_Y, svm1)
                print('a single train-test accuracy using doc2vec and svm with a linear kernel', accuracy1)
                svm2 = build_svm_classifier(train_X, train_Y, kernel='poly', gamma='scale')
                accuracy2 = estimate_svm_accuracy(test_X, test_Y, svm2)
                print('a single train-test accuracy using doc2vec and svm with a poly (4) kernel', accuracy2)
                svm3 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale')
                accuracy3 = estimate_svm_accuracy(test_X, test_Y, svm3)
                print('a single train-test accuracy using doc2vec and svm with a gaussian kernel', accuracy3)
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
    X, Y, = get_bow_data(development_data, min_count=5, max_frac=0.5)
    imdb_reviews = get_reviews(imdb_data_folder, imdb_sentiments, subfolders)
    train_X, train_Y, test_X, test_Y = get_train_test_split(0.7, X, Y)
    training_data, val_data = get_train_test_split(0.9, development_data)
    '''
    acc1 = cross_validate_svm(X, Y, kernel='rbf', gamma='scale', C=4.7)
    print('Cross validated accuracy when using a gaussian kernel', acc1)
    acc2 = cross_validate_svm(X, Y, kernel='linear', C=8.3)
    print('Cross validated accuracy when using a linear kernel', acc2)
    '''
    svm1 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale')
    svm2 = build_svm_classifier(train_X, train_Y, kernel='linear', gamma='scale')  
    for i in range(3):
        perm_p = run_permutation_test_on_two_svm_kernels(test_X, test_Y, svm1, svm2)
        perm_p2 = gaussian_vs_linear_permutation_test(X, Y)
        print(perm_p)
        print(perm_p2)

if __name__ == '__main__':
    main()