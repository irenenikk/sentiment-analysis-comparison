import re
import os 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from scipy import spatial

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

def build_doc2vec_model(reviews, vec_size, window_size, min_count, epochs, dm, dm_mean, pretrained=True, save=True):
    fname = get_tmpfile('doc2vec_{}_{}_{}'.format(vec_size, window_size, min_count))
    # BUG: distinguishing between pretrained models
    if pretrained and os.path.exists(fname):
        print('Loaded trained Doc2Vec from', fname)
        return Doc2Vec.load(fname)
    print('Training a Doc2Vec model for reviews')
    documents = [TaggedDocument(doc_tokenize(doc), [i]) for i, doc in enumerate(reviews)]
    model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=min_count, workers=4, dm=dm, epochs=epochs, dm_mean=dm_mean)
    if save:
        print('Saving to', fname)
        model.save(fname)
    return model

def train_doc2vec_model(review_data, train_frac=0.7, vec_size=100, window_size=4, min_count=5, epochs=30, dm=1, dm_mean=0):
    """ Train a doc2vec model from given text data and return the trained model and training set vectors. """
    model = build_doc2vec_model(review_data['review'].values, vec_size=vec_size, window_size=window_size, min_count=min_count, dm=dm, epochs=epochs, dm_mean=dm_mean, pretrained=False, save=False)
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
    support_vectors = svm.support_vectors_
    false_distances = [spatial.distance.cosine(supp, vec) for vec in test_X[false_mask] for supp in support_vectors]
    true_distances = [spatial.distance.cosine(supp, vec) for vec in test_X[~false_mask] for supp in support_vectors]
    sv_dist_from_misclassified = np.mean(false_distances), np.var(false_distances)
    sv_dist_from_others = np.mean(true_distances), np.var(true_distances)
    print('sv_dist_from_misclassified', sv_dist_from_misclassified)
    print('sv_dist_from_others', sv_dist_from_others)
