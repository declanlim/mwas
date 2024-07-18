from os import getpid
from psutil import Process
import tracemalloc
import time
from multiprocessing import Manager, Pool, cpu_count

import numpy as np
from scipy.stats import permutation_test


# Function to perform a single test
def mean_diff_statistic(x, y, axis):
    """if -ve, then y is larger than x"""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def get_process_memory():
    process = Process(getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size (RSS) in bytes


def perform_test(shared_dict, test_id, size=800):
    # Generate 2 lists of random elements for permutation test
    tracemalloc.start()
    data1, data2 = np.random.rand(size // 2), np.random.rand(size // 2)

    # Start measuring memory before computation
    start_memory = get_process_memory()

    # Perform permutation test
    result = permutation_test((data1, data2), mean_diff_statistic, n_resamples=10000, vectorized=True)

    # Measure memory after computation
    end_memory = get_process_memory()
    memory_used = end_memory - start_memory
    _, mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()


    shared_dict[test_id] = {'memory_start': start_memory, 'memory_end': end_memory, 'data1': data1, 'data2': data2, 'trace': mem}
    return shared_dict[test_id]


if __name__ == "__main__":
    process = Process()
    initial_memory_psutil = process.memory_info().rss
    initial_memory_percent = process.memory_percent()

    time_start = time.time()
    num_tests = 200
    num_workers = min(cpu_count(), num_tests)
    print(f"num workers: {num_workers}")

    # Create a manager for shared memory
    with Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary

        with Pool(processes=num_workers) as pool:
            results = pool.starmap(perform_test, [(shared_dict, i) for i in range(num_tests)])

    print("Time taken:", time.time() - time_start)
    ##print("Average memory usage:", np.mean([res['memory_end'] for res in results]))
    print("Max memory usage:", max([res['memory_end'] for res in results]))
    print([(res['result'], res['memory_end']) for res in results])

    print(f"max traced: {max([res['trace'] for res in results])}")

    # # Get final memory usage with psutil
    # final_memory_psutil = process.memory_info().rss
    # final_memory_percent = process.memory_percent()
    #
    # # Calculate maximum resident set size (RSS) memory usage during execution
    # memory_info = process.memory_info()
    # max_rss = memory_info.rss
    #
    # print(f"Initial RSS: {initial_memory_psutil}")
    # print(f"Final RSS: {final_memory_psutil}")
    # print(f"Memory Used: {(final_memory_psutil - initial_memory_psutil)}")
    # print(f"Maximum RSS: {max_rss}")
    # print(f"Initial Memory Percent: {initial_memory_percent} %")
    # print(f"Final Memory Percent: {final_memory_percent} %")

    # Running them in series to compare
    time_start = time.time()
    new_results = []
    for i in range(num_tests):
        new_results.append(perform_test({}, i))

    print("Time taken (serial):", time.time() - time_start)
