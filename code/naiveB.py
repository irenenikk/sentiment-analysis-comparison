from ngram_utils import get_ngram_probabilities
import numpy as np
from ngram_utils import get_bigram_list, get_uni_and_bi_grams, unigram_tokenize, bigram_tokenize, get_fraction_of_sentiment

class NaiveB:
        
    def __init__(self, classes, training_data_files, smooth, unigrams=None, bigrams=None, lowercase=True):
        if unigrams is None and bigrams is None:
            raise ValueError('Please give unigrams, bigrams or both')
        self.unigrams = unigrams
        self.bigrams = bigrams
        if lowercase:
            if unigrams is not None:
                self.unigrams['ngram'] = unigrams['ngram'].apply(lambda x: x.lower())
            if bigrams is not None:
                self.bigrams['ngram'] = bigrams['ngram'].apply(lambda x: x.lower())
        self.classes = classes
        self.smooth = smooth
        self.uni_class_conditioned_counts = {}
        self.bi_class_conditioned_counts = {}
        self.lowercase = lowercase
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
            prior = get_fraction_of_sentiment(training_data, cl)
            self.class_priors[cl] = prior


    def __str__(self):
        using_unigrams = self.unigrams is not None
        using_bigrams = self.bigrams is not None
        return f"A{'n un' if self.smooth == False else ' '}smoothed Naive Bayes classifier using {'unigrams and bigrams' if (using_bigrams and using_unigrams) else 'unigrams' if using_unigrams else 'bigrams'}"
        
    def in_training_data(self, feature, class_conditioned_counts):
        for counts in class_conditioned_counts.values():
            if feature in counts.index:
                return True
        return False 

    def get_class_probabilites(self, text, class_conditioned_counts, smooth_denom, tokenize):
        """ Return the probabilities for each class using the given data. """
        class_probs = np.zeros(len(self.classes))
        for i, cl in enumerate(self.classes):
            p = 0
            for feature in tokenize(text, lowercase=self.lowercase):
                # ignore features which are unseen in the training data, according to
                # http://web.stanford.edu/~jurafsky/slp3/4.pdf
                if not self.in_training_data(feature, class_conditioned_counts):
                    continue
                if feature in class_conditioned_counts[cl].index:
                    p += np.log(class_conditioned_counts[cl].loc[feature])
                # apply smoothing separately if feature not present in class
                elif self.smooth:
                    p += np.log(1/(smooth_denom))
                else:
                # this is to account for log(0), when a feature is not present in one class
                    p += -np.inf
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
        if (class_probs == class_probs[0]).all():
            # if all probabilities are the same, choose randomly
            return self.classes[np.random.randint(len(self.classes))]
        return self.classes[np.argmax(class_probs)]    
