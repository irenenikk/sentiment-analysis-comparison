from scipy.stats import binom
import numpy as np
import math

def calculate_p_value(N, k, q):
    """ Calculates p-value using the binomial distribution. """
    res = 0
    for i in range(k+1):
        res += binom.pmf(i, N, q)
    return 2*res

def sign_test_p_value(plus, minus, null):
    """ Calculates p-value for a two-tailed sign-test. """
    N = plus+minus
    k = min(plus, minus)
    return calculate_p_value(N, k, 0.5)

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
    return sign_test_p_value(plus, minus, null)

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
    return sign_test_p_value(plus, minus, null)

def get_accuracy(preds, Y):
   return (preds == Y).sum()/len(preds) 

def swap_randomly(A, B, shift_indx=None):
    assert len(A) == len(B), 'Both lists have to be the same size.'
    if shift_indx is None:
        # sample random amount of indexes to swap using a coin toss
        shift_amount = np.random.binomial(len(A), 0.5)
        indexes = list(range(len(A)))
        # flip the values in lists in random places
        index_perm = np.random.permutation(indexes)
        shift_indx = index_perm[:shift_amount]
    tmp = A[shift_indx]
    A[shift_indx] = B[shift_indx]
    B[shift_indx] = tmp
    return A, B

def permutation_test(true_labels, results_A, results_B, R=5000):
    """ Monte carlo permutation test on two different prediction lists. """
    acc_differences = np.zeros(R)
    true_acc_A = get_accuracy(results_A, true_labels)
    true_acc_B = get_accuracy(results_B, true_labels)
    true_diff = np.abs(true_acc_A - true_acc_B)
    for i in range(R):
        shuff_A, shuff_B = swap_randomly(results_A, results_B)
        acc_A = (shuff_A == true_labels).sum()/len(shuff_A)
        acc_B = (shuff_B == true_labels).sum()/len(shuff_B)
        acc_differences[i] = np.abs(acc_A - acc_B)
    return ((acc_differences >= true_diff).sum()+1)/(R+1)

def sample_variance(data):
    """ Calculate sample variance from data. """
    mean = np.mean(data)
    return np.sum(np.square(data-mean))

def get_mask(length, indexes):
    """ Turn indices into a boolean mask. """
    mask = np.zeros(length, dtype=bool)
    mask[indexes] = True
    return mask

def train_test_split_indexes(data, train_frac):
    """ Returns a random index split into two. """
    data_len = data.shape[0]
    indexes = list(range(data_len))
    rand_perm = np.random.permutation(indexes)
    train_batch = math.floor(data_len*train_frac)
    return rand_perm[:train_batch], rand_perm[train_batch:]

def get_train_test_split(train_frac, X, Y=None):
    """ Returns a random split of data into two. If Y is given, it is assumed to be labels for X. """
    train_indexes, test_indexes = train_test_split_indexes(X, train_frac)
    if Y is None:
        return X.iloc[train_indexes], X.iloc[test_indexes]
    return X[train_indexes], Y[train_indexes], X[test_indexes], Y[test_indexes]
