import re
import gensim
from gensim.test.utils import common_texts
import os
import pandas as pd
from sklearn import svm
from joblib import dump, load
from doc_utils import doc_tokenize, get_reviews, train_doc2vec_model, get_doc2vec_data, get_bow_vectors
from science_utils import sample_variance, get_mask, get_train_test_split, train_test_split_indexes
from ngram_utils import get_uni_and_bi_grams
import time
import numpy as np
from practical1 import build_data
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from scipy import spatial

models_file = 'models'

def build_svm_classifier(X, Y, pretrained=True, save=True, **kwargs):
    idd = '_'.join([f'{key}={str(value)}' for key, value in kwargs.items()])
    model_path = os.path.join(models_file, f'svm_{idd}_shape={X.shape[0]}_{X.shape[1]}.joblib')
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

def estimate_accuracy(X, Y, svm):
    """ Estimate the accuracy over test dataset using given unigrams and bigrams """
    preds = svm.predict(X)
    return (preds == Y).sum()/len(preds)

def cross_validate_svm(X, Y, folds=10, **kwargs):
    """ Run n-fold cross-validation using SVM. """
    start = time.time()
    indx = np.arange(0, len(X), folds)
    scores = np.zeros(folds)
    # separate blind test set
    for f in range(folds):
        test_mask = get_mask(len(X), indx+f)
        test_X, test_Y = X[test_mask], Y[test_mask]
        train_X, train_Y = X[~test_mask], Y[~test_mask]
        svm = build_svm_classifier(train_X, train_Y, pretrained=False, save=False, **kwargs)
        acc = estimate_accuracy(test_X, test_Y, svm)
        scores[f] = acc
    return np.mean(scores), sample_variance(scores)

def get_bow_data(review_data, train_frac=0.7, min_count=10, max_frac=0.5, dim=100):
    """ Get BOW vector training and test sets. """
    X = get_bow_vectors(review_data['review'].values, min_count, max_frac)
    Y = review_data['sentiment'].to_numpy()
    print('Created a BOW vector of shape', X.shape)
    return get_train_test_split(0.7, X, Y)
    
def visualize_vectors(X, Y, window_size, epochs):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig = go.Figure(data=go.Scatter(x=X_embedded[:,0],
                                    y=X_embedded[:,1],
                                    mode='markers',
                                    marker_color=Y))
    fig.update_layout(title=f'Doc2Vec t-SNE representation, window size {window_size}, epochs {epochs}')
    fig.show()

def evaluate_vector_qualities(X, Y, model):
    # 0: same label, 1: other vectors
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


def main():
    imdb_data_folder = 'aclImdb'
    imdb_sentiments = ['pos', 'neg']
    subfolders = ['train', 'test']
    reviews, _ = get_uni_and_bi_grams('data-tagged')
    review_data = build_data(reviews)
    training_data, blind_test = get_train_test_split(0.7, review_data, seed=42)
    train_X, train_Y, test_X, test_Y = get_bow_data(training_data, min_count=5, max_frac=0.5)
    print('Training with', train_X.shape[0], 'documents')
    svm2 = build_svm_classifier(train_X, train_Y, kernel='linear', pretrained=False)
    accuracy2 = estimate_accuracy(test_X, test_Y, svm2)
    print('Accuracy using bow and svm with a single train-test split and a linear kernel', accuracy2)
    svm5 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale', pretrained=False)
    accuracy5 = estimate_accuracy(test_X, test_Y, svm5)
    print('Accuracy using bow and svm with a single train-test split and a gaussian kernel', accuracy5)
    imdb_reviews = get_reviews(imdb_data_folder, imdb_sentiments, subfolders)

    maxim = -1
    max_params = {}
    for window_size in [2, 4, 6, 8]:
        for epochs in [15, 20, 25]:
            for dm in [0, 1]:
                print('-----------------------------------------')
                print('window size', window_size, 'epochs', epochs, 'dm', dm)
                train_X, train_Y, model_imdb = train_doc2vec_model(imdb_reviews, epochs=epochs, window_size=window_size, dm=dm)
                #visualize_vectors(train_X, train_Y, window_size, epochs)
                test_X = get_doc2vec_data(blind_test['review'].values, model_imdb)
                test_Y = blind_test['sentiment'].values
                visualize_vectors(test_X, test_Y, window_size, epochs)
                print('For test set vectors inferred with doc2vec')
                evaluate_vector_qualities(test_X, test_Y, model_imdb)
                svm1 = build_svm_classifier(train_X, train_Y, kernel='linear', pretrained=False)
                accuracy1 = estimate_accuracy(test_X, test_Y, svm1)
                print('a single train-test accuracy using doc2vec and svm with a linear kernel', accuracy1)
                svm2 = build_svm_classifier(train_X, train_Y, kernel='poly', gamma='scale', pretrained=False)
                accuracy2 = estimate_accuracy(test_X, test_Y, svm2)
                print('a single train-test accuracy using doc2vec and svm with a poly (4) kernel', accuracy2)
                svm3 = build_svm_classifier(train_X, train_Y, kernel='rbf', gamma='scale', pretrained=False)
                accuracy3 = estimate_accuracy(test_X, test_Y, svm3)
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

if __name__ == '__main__':
    main()