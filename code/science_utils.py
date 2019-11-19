from scipy.stats import binom
import numpy as np
import math

def calculate_p_value(N, k, q):
    """ Calculates p-value for a two-tailed sign-test. """
    res = 0
    for i in range(k+1):
        res += binom.pmf(i, N, q)
    return 2*res

def sign_test_lists(listA, listB):
    """ A list-based sign test. """
    assert len(listA) == len(listB), 'The lists should have the same amount of data points.'
    plus, minus, null = 0, 0, 0
    for i in range(len(listA)):
        a = listA[i]
        b = listB[i]
        if a == b:
            null += 1
        elif a > b:
            plus += 1
        else:
            minus += 1
    N = 2*math.ceil(null/2)+plus+minus
    k = math.ceil(null/2)+min(plus, minus)
    return calculate_p_value(N, k, 0.5)

def sign_test_systems(test_data, system_A, system_B):
    """ Returns the p-value of a two-tailed sign-test comparing two prediction systems."""
    plus, minus, null = 0, 0, 0
    for i in range(len(test_data)):
        inp = test_data['review'].iloc[i]
        a = system_A.predict(inp)
        b = system_B.predict(inp)
        true_label = test_data['sentiment'].iloc[i]
        if a == b:
            null += 1
        elif true_label == a:
            plus += 1
        elif true_label == b:
            minus += 1
    N = 2*math.ceil(null/2)+plus+minus
    k = math.ceil(null/2)+min(plus, minus)
    return calculate_p_value(N, k, 0.5)

def sample_variance(data):
    """ Calculate sample variance from data. """
    mean = np.mean(data)
    return np.sum(np.square(data-mean))

def get_mask(length, indexes):
    """ Turn indices into a boolean mask. """
    mask = np.ones(length, dtype=bool)
    mask[indexes] = False
    return mask

def train_test_split_indexes(data, train_frac, seed=0):
    """ Returns a random index split into two. """
    np.random.seed(seed)
    data_len = data.shape[0]
    indexes = list(range(data_len))
    rand_perm = np.random.permutation(indexes)
    train_batch = math.floor(data_len*train_frac)
    return rand_perm[:train_batch], rand_perm[train_batch:]

def get_train_test_split(train_frac, X, Y=None, seed=0):
    """ Returns a random split of data into two. If Y is given, it is assumed to be labels for X. """
    train_indexes, test_indexes = train_test_split_indexes(X, train_frac)
    if Y is None:
        return X.iloc[train_indexes], X.iloc[test_indexes]
    return X[train_indexes], Y[train_indexes], X[test_indexes], Y[test_indexes]