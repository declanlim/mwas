"""turn a bioproject metadata file into a
"""
import pandas as pd
import time
import os


def metadata_to_set_accession(metadata_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """takes a metadata dataframe fro a bioproject (accessed by biosample_id)
        and returns a list of biosample accessions and a dataframe where there are 3 columns:
        attribute(s), value(s) (an attribute(s) + value(s) pair makes a distinct set), and biosample_id,
        but biosample_id entries are a list of indexes which refer to which biosamples have that attribute(s) + value(s) pair.
        This will also filter out columns from the metadata dataframe that are the same value for all rows, or
        columns where all rows have unique values.
    """
    binary_key_coding = True  # if True, we'll use binary coding for the biosample_id, otherwise we'll use string coding
    n, m = metadata_df.shape  # n rows, m columns
    metadata_df.sort_values(by='biosample_id', inplace=True)
    new_df_builder = {}  # {biosample_vector: (attributes, values)} where attributes and values are in form "str; str; ..."

    for col in metadata_df.columns:
        col = col.replace(';', ':')  # to avoid confusion with the delimiter
        num_uniques = metadata_df[col].nunique()
        if col == 'biosample_id':
            continue
        elif num_uniques == 1 or num_uniques == n:
            metadata_df.drop(col, axis=1, inplace=True)
        else:
            metadata_df[col] = metadata_df[col].astype('category')
            factors = metadata_df[col].cat.categories

            for factor in factors:
                if pd.isna(factor) or factor == 'nan':
                    continue
                elif isinstance(factor, str):
                    factor = factor.replace(';', ',:')  # to avoid confusion with the delimiter

                biosample_vector_series = pd.Series(metadata_df[col] == factor)
                vector_len = sum(biosample_vector_series)
                if vector_len == 1:
                    continue
                include = vector_len < n / 2  # to save space, if most are true, we'll store all the false indexes, otherwise we'll store all the true indexes
                # note, use biosample_index_list in the final new_df, since biosample_vector can be too large an integer for pandas
                # biosample_code is for key value in new_df_builder (remember, you can't use a list as a key in a dictionary)

                try:
                    biosample_index_list = [i for i, x in enumerate(biosample_vector_series) if (x if include else not x)]
                except OverflowError:
                    print("Dataframe is too large to process. Exiting...")
                    return [], pd.DataFrame()
                if binary_key_coding:
                    try:
                        biosample_code = int(''.join(['1' if x else '0' for x in biosample_vector_series]), 2)  # this is faster than using str(biosample_vector_series)
                    except OverflowError:  # resorting to string coding
                        binary_key_coding = False
                        biosample_code = str(biosample_vector_series.tolist())
                else:
                    biosample_code = str(biosample_vector_series.tolist())

                # incorporate our set into the new_df_builder
                if biosample_code in new_df_builder:
                    curr_attr_val_pair = new_df_builder[biosample_code][0]
                    new_df_builder[biosample_code][0] = (f'{curr_attr_val_pair[0]}; {col}', f'{curr_attr_val_pair[1]}; {factor}')
                else:
                    new_df_builder[biosample_code] = [(col, factor), (biosample_index_list, include)]  # if mostly_true, we want to mark this index list as an exlude list

    new_df_data = []
    for entry in new_df_builder:
        (attributes, values), (index_list, include) = new_df_builder[entry]
        new_df_data.append({
            'attributes': attributes, 'values': values, 'biosample_index_list': index_list, 'include?': include
        })
    new_df = pd.DataFrame(new_df_data, columns=['attributes', 'values', 'biosample_index_list', 'include?'])

    biosamples_ref_lst = metadata_df['biosample_id'].tolist()
    return biosamples_ref_lst, new_df


if __name__ == '__main__':
    metadata_file = 'tests/metadata_set_maker_tests/TEST--PRJEB37099.csv'
    metadata_dataframe = pd.read_csv(metadata_file)
    start_time = time.time()
    biosamples_ref, set_df = metadata_to_set_accession(metadata_dataframe)
    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')
    # convert to csv store in csvs
    if os.path.exists('tests/TEST--PRJEB37099_SETS.csv'):
        os.remove('tests/TEST--PRJEB37099_SETS.csv')
    set_df.to_csv('../csvs/TEST--PRJEB37099_SETS.csv', index=False)
