"""turn a bioproject metadata file into a
"""
import sys
from typing import Any
import pandas as pd
import time
import os

from pandas import DataFrame


# TODO: everything to lowercase, and remove commas
def metadata_to_set_accession(metadata_df: pd.DataFrame, update_metadata_df=False) -> tuple[Any, DataFrame, str, DataFrame | tuple | Any, bool]:
    """takes a metadata dataframe from a bioproject (accessed by biosample_id)
        and returns a list of biosample accessions and a dataframe where there are 3 columns:
        attribute(s), value(s) (an attribute(s) + value(s) pair makes a distinct set), and biosample_id,
        but biosample_id entries are a list of indexes which refer to which biosamples have that attribute(s) + value(s) pair.
        This will also filter out columns from the metadata dataframe that are the same value for all rows, or
        columns where all rows have unique values.
    """
    comment = ''
    binary_key_coding = True  # if True, we'll use binary coding for the biosample_id, otherwise we'll use string coding
    n = metadata_df.shape[0]  # n rowsreturn [], pd.DataFrame(), "No rows in the metadata dataframe. "
    is_empty = n == 0
    metadata_df.sort_values(by='biosample_id', inplace=True)
    new_df_builder = {}  # {biosample_vector: (attributes, values)} where attributes and values are in form "str; str; ..."
    #
    # filter out rows with invalid biosample_ids
    if is_empty:
        comment += "The metadata was empty from the start. "
        print(comment)
    else:
        if metadata_df['biosample_id'].dtype != 'object':
            metadata_df['biosample_id'] = metadata_df['biosample_id'].astype(str)
        metadata_df = metadata_df[metadata_df['biosample_id'].str.startswith('SAM')]
        if metadata_df.shape[0] == 0:
            comment += "No valid biosample_ids found in the metadata dataframe. "
            print(comment)
            is_empty = True
        elif metadata_df.shape[0] < n:
            comment += f"Removed {n - metadata_df.shape[0]} rows with invalid biosample_ids. "
            print(comment)

    if not is_empty:
        for col in metadata_df.columns:
            num_uniques = metadata_df[col].nunique()
            if col == 'biosample_id':  # or col in blacklisted_fields:
                continue
            if num_uniques <= 1 or num_uniques == n:  # yes, it has been 0 strangely... (rare occurrence)
                continue  # metadata_df.drop(col, axis=1, inplace=True)  # dropping takes too long
            else:
                with pd.option_context('mode.chained_assignment', None):
                    metadata_df[col] = metadata_df[col].astype('category')
                factors = metadata_df[col].cat.categories

                for factor in factors:
                    if pd.isna(factor) or factor == 'nan':
                        continue

                    biosample_vector_series = pd.Series(metadata_df[col] == factor)
                    vector_len = sum(biosample_vector_series)
                    if vector_len == 1:
                        continue
                    include = vector_len < n / 2  # to save space, if most are true, we'll store all the false indexes, otherwise we'll store all the true indexes
                    # note, use biosample_index_list in the final new_df, since biosample_vector can be too large an integer for pandas
                    # biosample_code is for key value in new_df_builder (remember, you can't use a list as a key in a dictionary)

                    if isinstance(col, str):
                        col = col.replace(';', ':').lower()  # to avoid confusion with the delimiter
                    if isinstance(factor, str):
                        factor = factor.replace(';', ':').lower()  # to avoid confusion with the delimiter

                    try:
                        biosample_index_list = [i for i, x in enumerate(biosample_vector_series) if (x if include else not x)]
                    except OverflowError:
                        comment += "Data is too large to process. "
                        print(f"Data is too large to process. Exiting...")
                        return [], pd.DataFrame(), comment, _, True
                    if binary_key_coding:
                        try:
                            biosample_code = int(''.join(['1' if x else '0' for x in biosample_vector_series]), 2)  # this is faster than using str(biosample_vector_series)
                        except OverflowError:  # resorting to string coding
                            binary_key_coding = False
                            biosample_code = str(biosample_vector_series.tolist())
                            comment += "Resorted to using string coding for biosample_id due to an overflow error. "
                    else:
                        biosample_code = str(biosample_vector_series.tolist())

                    # incorporate our set into the new_df_builder
                    if biosample_code in new_df_builder:
                        curr_attr_val_pair = new_df_builder[biosample_code][0]
                        new_df_builder[biosample_code][0] = (f'{curr_attr_val_pair[0]}; {col}', f'{curr_attr_val_pair[1]}; {factor}')
                    else:
                        new_df_builder[biosample_code] = [(col, factor), (biosample_index_list, include)]  # if mostly_true, we want to mark this index list as an exlude list

    new_df_data = []
    biosamples_ref_lst = metadata_df['biosample_id'].tolist()
    for entry in new_df_builder:
        (attributes, values), (index_list, include) = new_df_builder[entry]
        if len(index_list) in (1, n - 1):  # ignore sets that are too small or too large, since we won't end up testing them in MWAS anyway
            continue
        if min(len(index_list), len(biosamples_ref_lst) - len(index_list)) < 4:
            test_type = 't-test'
        else:
            test_type = 'permutation-test'
        new_df_data.append({  # important to force values to be string - fixes a bug related to mixed datatypes in one column
            'attributes': attributes, 'values': str(values), 'biosample_index_list': index_list, 'include?': include, 'test_type': test_type
        })
    new_df = pd.DataFrame(new_df_data, columns=['attributes', 'values', 'biosample_index_list', 'include?', 'test_type'])
    empty_result = False
    if new_df.shape[0] == 0:
        comment += "No sets were generated from the metadata dataframe. "
        print("No sets were generated from the metadata dataframe.")
        empty_result = True

    return biosamples_ref_lst, new_df, comment, metadata_df if update_metadata_df else (), empty_result


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        metadata_file = 'tests/metadata_set_maker_tests/TEST--PRJEB37099.csv'
        metadata_dataframe = pd.read_csv(metadata_file)
        start_time = time.time()
        biosamples_ref, set_df, comment, _, _ = metadata_to_set_accession(metadata_dataframe)
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        # convert to csv store in csvs
        if os.path.exists('tests/TEST--PRJEB37099_SETS.csv'):
            os.remove('tests/TEST--PRJEB37099_SETS.csv')
        set_df.to_csv('tests/TEST--PRJEB37099_SETS.csv', index=False)
