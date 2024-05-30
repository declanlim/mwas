"""tests metadata_set_maker.py by comparing output to original metdata dataframe"""
import time
import traceback

import pandas as pd
import os
import sys

from main.metadata_set_maker import metadata_to_set_accession

METADATA_FILES = ['TEST--PRJEB37099.csv', 'TEST_LARGE--PRJDB11622.csv', 'TEST_MEDIUM--PRJDB10214.csv', 'TEST_SMALL--PRJDA67149.csv']
TEST_ALL_COLUMNS = True


# def test_very_large():
#     metadata_set_maker_test_setup(f'test_files/{METADATA_FILES[0]}')
#
#
# def test_large():
#     metadata_set_maker_test_setup(f'test_files/{METADATA_FILES[1]}')
#
#
# def test_medium():
#     metadata_set_maker_test_setup(f'test_files/{METADATA_FILES[2]}')
#
#
# def test_small():
#     metadata_set_maker_test_setup(f'test_files/{METADATA_FILES[3]}')
#
#
# def test_specific(metadata_file):
#     metadata_set_maker_test_setup(f'test_files/{metadata_file}')


def metadata_set_maker_test_setup(metadata_file):
    """tests metadata_set_maker.py by comparing output to original metdata dataframe
    """
    metadata_dataframe = pd.read_csv(metadata_file)
    create_time = time.time()
    biosamples_ref, set_df = metadata_to_set_accession(metadata_dataframe.copy())
    creation_time = time.time() - create_time

    out_file = f'test_outputs/{metadata_file.split("/")[-1][:-4]}_output.csv'
    if os.path.exists(out_file):
        os.remove(out_file)
    set_df.to_csv(out_file, index=False)
    output_size = os.path.getsize(out_file)

    if not TEST_ALL_COLUMNS:
        # spot checking random column
        row = set_df.sample().iloc[0]
        col = row['attributes'].split('; ')[0]
        set_df, values = reconstruct_metadata(set_df, biosamples_ref, col)
        return compare_metadata(set_df, metadata_dataframe, col, values), creation_time, output_size
    else:
        # testing all columns
        columns = set_df['attributes'].apply(lambda x: x.split('; ')[0]).unique()
        for col in columns:
            new_set_df, values = reconstruct_metadata(set_df, biosamples_ref, col)
            if not compare_metadata(new_set_df, metadata_dataframe, col, values):
                print(f"Failed on column: {col} on file: {metadata_file}")
                return False, creation_time, output_size
        return True, creation_time, output_size


def reconstruct_metadata(set_df, biosamples_ref, attr_name):
    """reconstruct a column metadata from set_df to compare with original metadata dataframe"""
    # set num rows to num of biosamples in biosamples_ref (generate empty cell rows for attribute values for now)
    new_df_constructor = []
    for biosample in biosamples_ref:
        new_df_constructor.append({
            'biosample_id': biosample,
            attr_name: ''
        })
    reconstructed_df = pd.DataFrame(new_df_constructor, columns=['biosample_id', attr_name])

    # getting values for this attribute (and their corresponding biosample indices
    values_for_this_attr = {}
    for _, row in set_df.iterrows():
        attributes = row['attributes'].split('; ')
        if attr_name in attributes:
            # row is splittable
            values = row['values']
            if isinstance(values, str):
                value = row['values'].split('; ')[attributes.index(attr_name)]
            else:
                value = values  # if it's a number
            values_for_this_attr[str(value)] = row['biosample_index_list'], row['include?']

    assert sum([len(biosample_index_list) for biosample_index_list in values_for_this_attr.values()]) <= len(biosamples_ref)

    # fill in the data frame attr_name column with values
    for value, (biosample_index_list, include) in values_for_this_attr.items():
        if not include:
            new_biosample_index_list = [index for index in range(len(biosamples_ref)) if index not in biosample_index_list]
        else:
            new_biosample_index_list = biosample_index_list
        for index in new_biosample_index_list:
            biosample = biosamples_ref[index]
            reconstructed_df.loc[reconstructed_df['biosample_id'] == biosample, attr_name] = value

    return reconstructed_df, values_for_this_attr


def compare_metadata(reconstructed_df, metadata_df, col, values):
    """compares a column of metadata_df with a column of set_df to see if they match up"""
    for _, row in metadata_df.iterrows():

        biosample = row['biosample_id']
        value = str(row[col])
        if isinstance(value, str):
            value.replace(';', ':')
        if value not in values:
            # then it's possible it's a singleton or a missing value (nan), otherwise there's an issue.
            if pd.isna(value) or value == 'nan':
                continue
            else:
                # check frequency of value in metadata_df  (this convulated way is necessary because of the semilcolon delimiter issue...)
                freqs = metadata_df[col].value_counts()
                for val in set(metadata_df[col].values):
                    if pd.isna(val) or val == 'nan' or freqs[val] == 1:
                        continue
                    else:
                        if isinstance(val, str):
                            val = val.replace(';', ':')
                        if str(val) not in values:
                            return False
        else:
            set_val = reconstructed_df[reconstructed_df['biosample_id'] == biosample][col].values[0]
            if set_val != value:
                return False
    return True


if __name__ == '__main__':
    def single_test(file, iteration=None) -> tuple[bool, float, int, float]:
        """Runs a single test"""
        start_time = time.time()
        iter_indicator = f"@ {iteration} iterations: " if iteration is not None else ""
        try:
            status, creation_time, output_size = metadata_set_maker_test_setup(file)
            if status:
                print(iter_indicator + f"{file} passed successfully. Time taken: {time.time() - start_time} seconds")
                return True, creation_time, output_size, time.time() - start_time
            else:
                print(iter_indicator + f"{file} FAILED gracefully. Time taken: {time.time() - start_time} seconds")
                return False, creation_time, output_size, time.time() - start_time
        except Exception as e:
            print(iter_indicator + f"{file} FAILED with error: {e}")
            print(traceback.format_exc())
            return False, 0.0, 0, time.time() - start_time

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1.endswith('.csv'):
            single_test(arg1)
        elif arg1.endswith('.txt'):
            with (open(arg1, 'r') as f, open('failed.txt', 'w') as failed_f,
                  open('passed.txt', 'w') as passed_f, open('results.csv', 'w') as results_f):
                failed, passed = 0, 0
                iterations, total_creation_time = 1, 0.0
                results_f.write("bioproject, original_pickle_size, condensed_csv_size, creation_time, test_time, status\n")
                for line in f:
                    # line is in form <file_name then size (if applicable)>
                    info = line.split(' ')
                    test_file = info[0]
                    pickle_size = info[1] if len(info) > 1 else None
                    if test_file.endswith('.csv'):
                        status_, creation_time_, output_size_, test_time = single_test(test_file, iterations)
                        if status_:
                            passed_f.write(test_file + '\n')
                            passed += 1
                        else:
                            failed_f.write(test_file + '\n')
                            failed += 1
                        iterations += 1
                        total_creation_time += creation_time_

                        bioproject = test_file.split('/')[-1][:-4]
                        results_f.write(f"{bioproject},{pickle_size if pickle_size is not None else 'missing'},"
                                        f"{output_size_},{creation_time_},{test_time},{status_}\n")
                if failed > 0:
                    print(f"Failed {failed} bioprojects out of {failed + passed} bioprojects")
                else:
                    print(f"Passed all {passed} bioprojects")
                print(f"Total time used towards creating files: {total_creation_time} seconds,")
                print(f"which is an average of {total_creation_time / iterations} seconds per iteration")
