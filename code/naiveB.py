from ngram_utils import get_ngram_probabilities
import numpy as np
from ngram_utils import get_bigram_list, get_uni_and_bi_grams, unigram_tokenize, bigram_tokenize

class NaiveB:
        
    def __init__(self, classes, training_data_files, smooth, unigrams=None, bigrams=None):
        if unigrams is None and bigrams is None:
            raise ValueError('Please give unigrams, bigrams or both')
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.classes = classes
        self.smooth = smooth
        self.uni_class_conditioned_counts = {}
        self.bi_class_conditioned_counts = {}
        # prepare conditional probabilities here to make prediction faster
        self.class_priors = {}
        for cl in self.classes:
            training_data = None
            if self.unigrams is not None:
                training_unigrams = self.unigrams[self.unigrams['file_no'].isin(training_data_files)]
                self.uni_class_conditioned_counts[cl] = get_ngram_probabilities(training_unigrams, cl, smooth)
                self.uni_smooth_denom = (training_unigrams['sentiment']==cl).sum()+training_unigrams['ngram'].nunique()
                training_data = unigrams
            if self.bigrams is not None:
                training_bigrams = self.bigrams[self.bigrams['file_no'].isin(training_data_files)]
                self.bi_class_conditioned_counts[cl] = get_ngram_probabilities(training_bigrams, cl, smooth)
                self.bi_smooth_denom = (training_bigrams['sentiment']==cl).sum()+training_bigrams['ngram'].nunique()
                training_data = training_bigrams
            sentiment_files = training_data[['file_id', 'sentiment']].groupby('file_id').agg(lambda x: x.value_counts().index[0])
            prior = (sentiment_files['sentiment'] == cl).sum()/len(sentiment_files)
            self.class_priors[cl] = prior


    def __str__(self):
        using_unigrams = self.unigrams is not None
        using_bigrams = self.bigrams is not None
        return f"Naive Bayes classifier using {'unigrams and bigrams' if (using_bigrams and using_unigrams) else 'unigrams' if using_unigrams else 'bigrams'}"
        
    def get_class_probabilites(self, text, class_conditioned_counts, smooth_denom, tokenize):
        """ Return the probabilities for each class using the given data. """
        class_probs = np.zeros(len(self.classes))
        for i, cl in enumerate(self.classes):
            p = 0
            for word in tokenize(text):
                if word in class_conditioned_counts[cl].index:
                    p += np.log(class_conditioned_counts[cl].loc[word])
                # apply smoothing separately if word not present in class
                elif self.smooth:
                    p += np.log(1/(smooth_denom))
            # the prior is the fraction of documents in a specific class
            # the mean is just to have one value for each file_id. The sentiment is the same for all reviews in one file.
            p += np.log(self.class_priors[cl])
            class_probs[i] = p
        return class_probs    
        
    def predict(self, text):
        # set the binary classification labels
        class_probs = np.zeros(len(self.classes))
        if self.unigrams is not None:
            class_probs += self.get_class_probabilites(text, self.uni_class_conditioned_counts, self.uni_smooth_denom, unigram_tokenize)
        if self.bigrams is not None:
            class_probs += self.get_class_probabilites(text, self.bi_class_conditioned_counts, self.bi_smooth_denom, bigram_tokenize)
        return self.classes[np.argmax(class_probs)]    
