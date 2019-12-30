import re
import os 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from scipy import spatial
from sklearn.metrics.pairwise import cosine_distances

def doc_tokenize2(doc):
    """ Split document into words by keeping punctuation. """
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    doc = re.sub(r"([?.!,])", r" \1 ", doc)
    doc = w = re.sub(r'[" "]+', " ", doc)
    return doc.split()

def doc_tokenize(doc):
    """ Split document into lowercased words removing any special characters. """
    return [x.lower() for x in re.sub(r'[^a-zA-Z\s]', '', doc).split()]

def get_reviews(imdb_data_folder, imdb_sentiments, subfolders):
    """ Return a dataframe of loaded reviews with grade and sentiment. """
    print('Getting reviews from', imdb_data_folder)
    # using lists is ugly but faster than appending to a dataframe
    review_list = []
    review_id_list = []
    review_grade_list = []
    review_sentiment_list = []
    for sent in imdb_sentiments:
        for subf in subfolders:
            for review_file in os.listdir(os.path.join(imdb_data_folder, subf, sent)):
                idd = review_file.split('_')[0]
                review_id_list += [idd]
                grade = re.search('_(.*)\.txt', review_file).group(1)
                review_grade_list += [grade]
                f = open(os.path.join(imdb_data_folder, subf, sent, review_file), 'r+')
                review = f.read()
                review_list += [review]
                review_sentiment_list += [1 if sent == 'pos' else -1]
    reviews = pd.DataFrame(list(zip(review_list, review_id_list, review_grade_list, review_sentiment_list)), columns=['review', 'id', 'grade', 'sentiment'])
    print('Found', len(reviews), ' IMDB reviews')
    return reviews

def build_doc2vec_model(reviews, vec_size, window_size, min_count, epochs, pretrained=False, save=False, **kwargs):
    idd = '_'.join(['{}={}'.format(key, str(value)) for key, value in kwargs.items()])
    fname = get_tmpfile('doc2vec_{}_{}_{}_{}_{}'.format(len(reviews), vec_size, window_size, min_count, idd))
    # BUG: distinguishing between pretrained models with random splits. Use with caution.
    if pretrained and os.path.exists(fname):
        print('Loaded trained Doc2Vec from', fname)
        return Doc2Vec.load(fname)
    print('Training a Doc2Vec model for reviews')
    documents = [TaggedDocument(doc_tokenize(doc), [i]) for i, doc in enumerate(reviews)]
    model = Doc2Vec(documents, vector_size=vec_size, window=window_size, min_count=min_count, workers=4, epochs=epochs, **kwargs)
    if save:
        print('Saving to', fname)
        model.save(fname)
    return model

def train_doc2vec_model(review_data, vec_size=100, window_size=4, min_count=4, epochs=30, dm=0, **kwargs):
    """ Train a doc2vec model from given text data and return the trained model and training set vectors. """
    model = build_doc2vec_model(review_data['review'].values, vec_size=vec_size, window_size=window_size, min_count=min_count, dm=dm, epochs=epochs, **kwargs)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    X = np.asarray([model.docvecs[i] for i in range(len(model.docvecs))])
    Y = review_data['sentiment'].to_numpy()
    return X, Y, model

def get_doc2vec_data(reviews, model):
    """ Get Doc2Vec vectors from a trained model. """
    return np.asarray([model.infer_vector(doc_tokenize(review)) for review in reviews])

def get_bow_vectors(review_data, min_count=5, max_frac=.5, vectorizer=None, lowercase=True, frequency=True, bigrams=False):
    """ Return BOW vectors for a given array of text. """
    if vectorizer is None:
        ngram_range = (2, 2) if bigrams else (1, 1)
        vectorizer = CountVectorizer(min_df=min_count, max_df=max_frac, lowercase=lowercase, ngram_range=ngram_range)
        vectorizer.fit(review_data)
    bows = vectorizer.transform(review_data).toarray()
    if frequency:
        return bows, vectorizer
    # if frequency is false only look at presence of feature
    bows[bows > 0] = 1
    return bows, vectorizer

def evaluate_vector_qualities(X, Y):
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

def calculate_vector_distances(X1, X2):
    return cosine_distances(X1, X2)

def model_error_analysis(test_X, test_data, svm):
    """ Print out misclassified instances with their prediction and true label. """
    preds = svm.predict(test_X)
    false_mask = (preds != test_data['sentiment'].values)
    false_indices = np.where(false_mask)[0]
    # choose 5 random indices and print
    indices = false_indices[np.random.choice(len(false_indices), size=10, replace=False)]
    for i in indices:
        print('Review')
        print(test_data['review'].iloc[i])
        print('True label', test_data['sentiment'].iloc[i])
        print('Classified as', preds[i])
        print('Confidence', svm.predict_proba([test_X[i]]))
        print('-----------------')
    # compare the vectors of the falsely classified instances to support vectors
    support_vectors = svm.support_vectors_
    false_confidences = [np.max(svm.predict_proba([vec])) for vec in test_X[false_mask]]
    true_confidences = [np.max(svm.predict_proba([vec])) for vec in test_X[~false_mask]]
    sv_misclassified_confidence = np.mean(false_confidences), np.var(false_confidences)
    sv_true_confidence = np.mean(true_confidences), np.var(true_confidences)
    print('sv_misclassified_confidence mean', sv_misclassified_confidence)
    print('sv_true_confidence mean', sv_true_confidence)
