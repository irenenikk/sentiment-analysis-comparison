from ngram_utils import preprocess_ngrams
import numpy as np
from ngram_utils import get_bigram_list, get_uni_and_bi_grams, unigram_tokenize, bigram_tokenize

class NaiveB:
        
    def __init__(self, classes, unigrams=None, bigrams=None):
        if unigrams is None and bigrams is None:
            raise ValueError('Please give unigrams, bigrams or both')
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.classes = classes

    def __str__(self):
        using_unigrams = self.unigrams is not None
        using_bigrams = self.bigrams is not None
        return f"Naive Bayes classifier using {'unigrams and bigrams' if (using_bigrams and using_unigrams) else 'unigrams' if using_unigrams else 'bigrams'}"
        
    def get_class_probabilites(self, text, tokenize, training_data, smooth, min_freq):
        class_probs = np.zeros(len(self.classes))
        for i, cl in enumerate(self.classes):
            p = 0
            conditioned_counts = preprocess_ngrams(training_data, cl, min_freq, smooth)
            for word in tokenize(text):
                if word in conditioned_counts.index:
                    p += np.log(conditioned_counts.loc[word])
                # apply smoothing separately if word not present in class
                elif smooth:
                    smooth_denom = (training_data['sentiment']==cl).sum()+training_data['ngram'].nunique()
                    p += np.log(1/(smooth_denom))
            # the prior is the fraction of documents in a specific class
            # the mean is just to have one value for each file_id. The sentiment is the same for all reviews in one file.
            sentiment_files = training_data[['file_id', 'sentiment']].groupby('file_id').mean()
            prior = (sentiment_files['sentiment'] == cl).sum()/len(sentiment_files)
            p += np.log(prior)
            class_probs[i] = p
        return class_probs    
        
    def predict(self, text, training_data_files, smooth=True):
        # set the binary classification labels
        class_probs = np.zeros(len(self.classes))
        if self.unigrams is not None:
            training_unigrams = self.unigrams[self.unigrams['file_no'].isin(training_data_files)]
            class_probs += self.get_class_probabilites(text, unigram_tokenize, training_unigrams, smooth, min_freq=4)
        if self.bigrams is not None:
            training_bigrams = self.bigrams[self.bigrams['file_no'].isin(training_data_files)]
            class_probs += self.get_class_probabilites(text, bigram_tokenize, training_bigrams, smooth, min_freq=7)
        return self.classes[np.argmax(class_probs)]    
