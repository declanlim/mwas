"""tests metadata_set_maker.py by comparing output to original metdata dataframe"""
import unittest
import time
import pandas as pd
import os

from main.metadata_set_maker import metadata_to_set_accession


class TestMetadataSetMaker(unittest.TestCase):
    """tests metadata_set_maker.py by comparing output to original metdata dataframe"""
    metadata_files = ['TEST--PRJEB37099.csv', 'TEST_LARGE--PRJDB11622.csv', 'TEST_MEDIUM--PRJDB10214.csv', 'TEST_SMALL--PRJDA67149.csv']
    test_all_columns = True

    def test_very_large(self):
        self.metadata_set_maker_test_setup(self.metadata_files[0])

    def test_large(self):
        self.metadata_set_maker_test_setup(self.metadata_files[1])

    def test_medium(self):
        self.metadata_set_maker_test_setup(self.metadata_files[2])

    def test_small(self):
        self.metadata_set_maker_test_setup(self.metadata_files[3])

    def metadata_set_maker_test_setup(self, metadata_file):
        metadata_dataframe = pd.read_csv(metadata_file)
        start_time = time.time()
        biosamples_ref, set_df = metadata_to_set_accession(metadata_dataframe.copy())
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')

        file = f'{metadata_file[:-4]}_output.csv'
        if os.path.exists(file):
            os.remove(file)
        set_df.to_csv(file, index=False)

        if not self.test_all_columns:
            # spot checking random column
            row = set_df.sample().iloc[0]
            col = row['attributes'].split('; ')[0]
            set_df, values = self.reconstruct_metadata(set_df, biosamples_ref, col)
            assert self.compare_metadata(set_df, metadata_dataframe, col, values)
        else:
            # testing all columns
            columns = set_df['attributes'].apply(lambda x: x.split('; ')[0]).unique()
            for col in columns:
                new_set_df, values = self.reconstruct_metadata(set_df, biosamples_ref, col)

                if not self.compare_metadata(new_set_df, metadata_dataframe, col, values):
                    print(f"Failed for {col}")
                    assert False, f"Failed for {col}"
            assert True

    def reconstruct_metadata(self, set_df, biosamples_ref, attr_name):
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
            if attr_name in row['attributes']:
                # row is splittable
                values = row['values']
                if isinstance(values, str):
                    value = row['values'].split('; ')[row['attributes'].split('; ').index(attr_name)]
                else:
                    value = values  # if it's a number
                values_for_this_attr[value] = row['biosample_index_list'], row['include?']

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

    def compare_metadata(self, reconstructed_df, metadata_df, col, values):
        """compares a column of metadata_df with a column of set_df to see if they match up"""
        for _, row in metadata_df.iterrows():
            biosample = row['biosample_id']
            value = row[col]
            if value not in values:
                # it's possible it's a singleton or a missing value (nan)
                if pd.isna(value) or value == 'nan':
                    continue
                else:
                    # get frequency of value in metadata_df
                    freq = metadata_df[col].value_counts()[value]
                    if freq == 1:
                        continue
                    else:
                        return False
            else:
                set_val = reconstructed_df[reconstructed_df['biosample_id'] == biosample][col].values[0]
                if set_val != value:
                    # try replacing the ; with ,:
                    if set_val != value.replace(';', ',:'):
                        return False
        return True
