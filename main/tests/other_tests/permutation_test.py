import numpy as np
import time
from itertools import combinations
from scipy.stats import permutation_test
# n choose k
from math import comb
import tracemalloc

def mean_diff_statistic(x, y, axis=0):
    """if -ve, then y is larger than x"""
    return abs(np.mean(x, axis=axis) - np.mean(y, axis=axis))


# Example data
# data1, data2 = [1, 2, 3, 4], [10, 11, 12, 13, 14, 16]
#data1, data2 = [0, 0, 1], [1, 2, 3, 4, 5, 0, 0]

data1 = [0, 1] * 69#, 62, 37, 84.1, 39, 18, 13, 44]
data2 = [22, 1] #, 0.1, 7, 5, 98, 22]


n = len(data1) + len(data2)
k = len(data1)
# Number of possible permutations
possible_permutations = comb(n, k)
print(f"Number of possible permutations: {possible_permutations}\n")

# first_time = time.time()
# # Combine datasets
# combined_data = np.array(data1 + data2)
#
# # Number of elements in each original group
# n1 = len(data1)
# n2 = len(data2)
#
# # Calculate the observed test statistic
# observed_statistic = mean_diff_statistic(data1, data2)
#
# # Generate all possible permutations
# all_permutations = list(combinations(combined_data, n1))
#
# # Calculate the distribution of test statistics under the null hypothesis
# null_distribution = []
# for perm in all_permutations:
#     group1 = np.array(perm)
#     group2 = np.array([x for x in combined_data if x not in group1])
#     null_distribution.append(mean_diff_statistic(group1, group2))
#
# # Calculate the p-value
# null_distribution = np.array(null_distribution)
# p_value = np.mean(null_distribution >= observed_statistic)
# print(f"Observed Statistic: {observed_statistic}")
# print(f"P-value from manual: {p_value}")
#
# first_time = time.time() - first_time
# print(f"Time taken manual: {first_time}")

time_difs = []
time_ratios = []
mem_difs = []
for i in range(50):
    tracemalloc.start()
    first_time = time.time()
    # Use the permutation_test function from scipy using dif of means as the test statistic
    res = permutation_test((data1, data2), statistic=mean_diff_statistic, n_resamples=10000, vectorized=True)
    print(f"P-value from scipy 10000: {res.pvalue}")

    first_time = time.time() - first_time
    print(f"Time taken scipy 10000: {first_time}")
    _, first_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory 10000: {first_mem}\n")

    # second
    tracemalloc.start()
    second_time = time.time()
    # Use the permutation_test function from scipy using dif of means as the test statistic
    res = permutation_test((data1, data2), statistic=mean_diff_statistic, n_resamples=min(10000, possible_permutations), vectorized=True)
    print(f"P-value from scipy LESS: {res.pvalue}")

    second_time = time.time() - second_time
    print(f"Time taken scipy LESS: {second_time}")
    _, second_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory LESS: {second_mem}\n")

    time_difs.append(second_time - first_time)
    time_ratios.append(first_mem / second_mem)
    mem_difs.append(first_mem - second_mem)

print(f"Average time difference: {np.mean(time_difs)}")
print(f"Average time ratio: {np.mean(time_ratios)}")
print(f"Average memory difference: {np.mean(mem_difs)}")
print(time_difs)
print(mem_difs)
