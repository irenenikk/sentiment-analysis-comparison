import re
import os 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

def doc_tokenize(doc):
    """ Split document into lowercased words removing any special characters. """
    return [x.lower() for x in re.sub(r'[^a-zA-Z\s]', '', doc).split()]

def get_reviews(imdb_data_folder, imdb_sentiments, subfolders):
    """ Return a dataframe of loaded reviews with grade and sentiment. """
    print('Getting reviews from', imdb_data_folder)
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
    print('Found', len(reviews), 'reviews')
    return reviews

def build_doc2vec_model(reviews, vec_size, window_size, min_count, epochs, dm, pretrained=True, save=True):
    fname = get_tmpfile(f'doc2vec_{vec_size}_{window_size}_{min_count}')
    if pretrained and os.path.exists(fname):
        print('Loaded trained Doc2Vec from', fname)
        return Doc2Vec.load(fname)
    print('Training a Doc2Vec model for reviews')
    documents = [TaggedDocument(doc_tokenize(doc), [i]) for i, doc in enumerate(reviews)]
    model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=min_count, workers=4, dm=dm, epochs=epochs)
    if save:
        print('Saving to', fname)
        model.save(fname)
    return model

def train_doc2vec_model(review_data, train_frac=0.7, vec_size=100, window_size=4, min_count=4, epochs=30, dm=1):
    """ Train a doc2vec model from given text data and return the trained model and training set vectors. """
    model = build_doc2vec_model(review_data['review'].values, vec_size=vec_size, window_size=window_size, min_count=min_count, dm=dm, epochs=epochs, pretrained=False, save=False)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    X = np.asarray([model.docvecs[i] for i in range(len(model.docvecs))])
    Y = review_data['sentiment'].to_numpy()
    return X, Y, model

def get_doc2vec_data(reviews, model):
    """ Get Doc2Vec vectors from a trained model. """
    return np.asarray([model.infer_vector(review.lower().split()) for review in reviews])

def get_bow_vectors(review_data, min_count, max_frac):
    """ Return BOW vectors for a given array of text. """
    vectorizer = CountVectorizer(min_df=min_count, max_df=max_frac)
    vectorizer.fit(review_data)
    return vectorizer.transform(review_data).toarray()
