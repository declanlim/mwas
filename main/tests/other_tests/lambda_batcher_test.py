"""tests bacthing lambdas.
e.g. total tests = 80, lambda_area = 30, n_permutation_sets = 16, groups = ['A', 'B', 'C', 'D', 'E']"""
import math


def f(lambda_area: int, n_permutation_sets: int, groups: list[str]) -> list[dict[str, tuple[int, int]]]:
    """tests bacthing lambdas."""
    num_tests_added = curr_group = left_off_at = 0
    jobs = []
    total_tests = len(groups) * n_permutation_sets
    for i in range(0, total_tests, lambda_area):
        # for each lambda job
        lambda_size = lambda_area if i + lambda_area < total_tests else total_tests - i
        focus_groups = {}
        started_at = left_off_at
        tests_added_to_lambda = 0
        while tests_added_to_lambda < lambda_size:
            tests_added_to_lambda += 1  # add a test
            left_off_at += 1  # record where we are
            num_tests_added += 1  # update where we are in a group relative to the total tests
            if num_tests_added % n_permutation_sets == 0:  # finished adding tests for a group
                focus_groups[groups[curr_group]] = (started_at, left_off_at)
                curr_group += 1
                started_at = left_off_at = 0
        if left_off_at > 0:
            focus_groups[groups[curr_group]] = (started_at, left_off_at)
        print(focus_groups)
        jobs.append(focus_groups)

    assert math.ceil(total_tests / lambda_area) == len(jobs)
    return jobs


def test_f():
    f(25, 23, ['A', 'B', 'C', 'D', 'E', 'F'])
