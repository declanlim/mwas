"""iterate through a list of mwas output csvs and extract certain data from them"""
import sys

import pandas as pd


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: mwas_results_analyze.py <mwas_output_files.txt>\n or: mwas_results_analyze.py <output.csv>")
        sys.exit(1)
    if sys.argv[1].endswith('.csv'):
        mwas_outputs_list = [sys.argv[1]]
    else:
        mwas_outputs = sys.argv[1]
        mwas_outputs_list = []
        with open(mwas_outputs, 'r') as f:
            for line in f:
                mwas_outputs_list.append(line.strip())
    results = ('bioproject,total_biosamples,number_tests,number_permutation_tests,number_metadata_sets,number_groups,num_skipped_groups,'
               'max_memory_usage,avg_memory_usage,avg_perms_memory_usage,total_memory_usage,'
               'max_runtime,avg_runtime,avg_perms_memory_usage,total_runtime\n')
    for mwas_output in mwas_outputs_list:
        df = pd.read_csv(mwas_output)
        # split the df into subsets by bioproject and then iterate over them such that each one writes a line to the results
        bioprojects = df['bioproject'].unique()

        for bioproject in bioprojects:
            df = df[df['bioproject'] == bioproject]

            # max, avg, total memory usage
            max_memory_usage = df['memory_usage_bytes'].max()
            avg_memory_usage = df['memory_usage_bytes'].mean()
            # filter for ones that have 'permutation' in status column
            avg_memory_usage_permutations = df[df['status'].str.contains('permutation')]['memory_usage_bytes'].mean()
            total_memory_usage = df['memory_usage_bytes'].sum()

            # max, avg, total runtime
            max_runtime = df['runtime_seconds'].max()
            avg_runtime = df['runtime_seconds'].mean()
            # filter for ones that have 'permutation' in status column
            avg_runtime_permutations = df[df['status'].str.contains('permutation')]['runtime_seconds'].mean()
            total_runtime = df['runtime_seconds'].sum()

            # number of biosamples - just take the sum of num_true and num_false on the first row
            total_biosamples = df['num_true'][0] + df['num_false'][0]

            # total number of groups. We take the 2nd column in df, since it's not always the named group
            number_groups = df[df.columns[1]].unique().shape[0]

            # number of metadata sets
            number_metadata_sets = df.shape[0] / number_groups

            # number skipped groups
            number_skipped_groups = df[df['status'].str.contains('skipped')].shape[0] / number_metadata_sets

            # total tests
            total_tests = df.shape[0] - number_skipped_groups * number_metadata_sets

            # percent of tests that are permutation tests
            num_permutations = df[df['status'].str.contains('permutation')].shape[0]

            results += f'{bioproject},{total_biosamples},{total_tests},{num_permutations},{number_metadata_sets},{number_groups},{number_skipped_groups},' \
                       f'{max_memory_usage},{avg_memory_usage},{avg_memory_usage_permutations},{total_memory_usage},' \
                       f'{max_runtime},{avg_runtime},{avg_runtime_permutations},{total_runtime}\n'
    try:
        with open('mwas_results_summary.csv', 'w') as f:
            f.write(results)
    except IOError:
        print("Error writing to file, check to make sure it isn't open in another program.")
    print('Done. Check mwas_results_summary.csv for the results.')
