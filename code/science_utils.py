from scipy.stats import binom
import numpy as np
import math

def calculate_p_value(N, k, q):
    res = 0
    for i in range(k+1):
        res += binom.pmf(i, N, q)
    return 2*res

def sign_test(test_data, system_A, system_B, n=10):
    plus, minus, null = 0, 0, 0
    for i in range(len(test_data)):
        inp = test_data['review'].iloc[i]
        a = system_A(inp)
        b = system_B(inp)
        true_label = test_data['sentiment'].iloc[i]
        if true_label == a:
            plus += 1
        elif true_label == b:
            minus += 1
        else:
            null += 1
    N = 2*math.ceil(null/2)+plus+minus
    k = math.ceil(null/2)+min(plus, minus)
    return calculate_p_value(N, k, 0.5)

def sample_variance(data):
    mean = np.mean(data)
    return np.sum(np.square(data-mean))
