"""Generalized MWAS
"""
# built-in libraries
import os
import pickle
import warnings
# import logging
import sys

# import required libraries
import pandas as pd
import random
import numpy as np
import scipy.stats as stats
import psycopg2

OUTPUT_DIR_DISK = 'temp/outputs'
PICKLE_DIR = 'temp/pickles'

MAP_UNKNOWN = 0  # maybe np.nan
BLOCK_SIZE = 1000
NORMALIZING_CONST = 1000000
SCHEDULING = False
BLACKLIST = []  # list of bioprojects that are too large
OUT_COLS = ['status', 'metadata_field', 'metadata_value', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
            'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value']
CONNECTION_INFO = {
    'host': 'serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com',
    'database': 'summary',
    'user': 'public_reader',
    'password': 'serratus'
}
query = """
    SELECT bio_project, run, bio_sample, spots
    FROM %s -- srarun
    WHERE run in (%s)
"""


class BioProjectInfo:
    """Class to store information about a bioproject"""

    def __init__(self, name: str, metadata_pickle_path: str, metadata_size: int = None) -> None:
        self.name = name
        self.metadata_path = metadata_pickle_path
        self.metadata_size = metadata_size
        self.metadata_df = None
        self.metadata_row_count = None

    def get_metadata_size(self) -> int:
        """Get the size of the metadata pickle file, or set it if it's not set yet
        """
        if self.metadata_size is None:
            self.metadata_size = os.path.getsize(self.metadata_path)
        return self.metadata_size

    def load_metadata(self) -> pd.DataFrame | None:
        """load metadata pickle file into memory as a DataFrame.
        Note: we should only use this if we're currently working with the bioproject
        """
        try:
            with open(self.metadata_path, 'rb') as f:
                self.metadata_df = pickle.load(f)
                self.metadata_row_count = self.metadata_df.shape[0]
                # can't we just use pd.read_pickle(self.metadata_path)?
                return self.metadata_df
        except Exception as e:
            print(f"Error in loading metadata for {self.name}")
            print(e)

    def delete_metadata(self):
        """Delete the metadata pickle file, and the dataframe from memory
        Note: this should only be used when we're done with the bioproject
        """
        del self.metadata_df
        self.metadata_df = None
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self.metadata_path = None


def get_bioprojects_df(runs: list) -> pd.DataFrame | None:
    """Get the bioproject data for the given runs
    """
    try:
        conn = psycopg2.connect(**CONNECTION_INFO)
        conn.close()  # close the connection because we are only checking if we can connect
        print(f"Successfully connected to database at {CONNECTION_INFO['host']}")
    except psycopg2.Error:
        print(f"Unable to connect to database at {CONNECTION_INFO['host']}")

    runs_str = ", ".join([f"'{run}'" for run in runs])

    with psycopg2.connect(**CONNECTION_INFO) as conn:
        try:
            df = pd.read_sql(query % ('srarun', runs_str), conn)
            df['spots'] = df['spots'].replace(0, NORMALIZING_CONST)
            return df
        except psycopg2.Error:
            print("Error in executing query")
            return None


def metadata_retrieval(biopj_block: list[str]) -> dict[str, BioProjectInfo]:
    """Retrieve metadata for the given bioprojects, and organize them in a dictionary
    """
    # TODO: a constant for the blacklist of projects that are too large (manually calculate this beforehand)
    # make a single s5cmd call to download all the metadata files in one request (they should get loaded to mnt)
    import subprocess
    import os

    def is_tmpfs_mounted(mount_point):
        with open('/proc/mounts', 'r') as f:
            mounts = f.readlines()
        for mount in mounts:
            parts = mount.split()
            if parts[1] == mount_point and parts[2] == 'tmpfs':
                return True
        return False

    # Configuration
    s3_bucket = 's3://serratus-biosamples/mwas_setup/'
    file_prefix = 'your/prefix/'  # Change as needed
    max_file_size = 50 * 1024 * 1024  # Maximum file size in bytes (e.g., 50 MB)
    tmpfs_dir = '/mnt/tmpfs'
    file_list = biopj_block

    # Check if /tmp is mounted as tmpfs, otherwise use /mnt/tmpfs
    if is_tmpfs_mounted('/tmp'):
        tmp_dir = '/tmp'
        print("/tmp is mounted as tmpfs (RAM)")
    else:
        tmp_dir = tmpfs_dir
        print("/tmp is not mounted as tmpfs (RAM), using /mnt/tmpfs")
        os.makedirs(tmpfs_dir, exist_ok=True)
        # Mount tmpfs without specifying size
        subprocess.run(f"sudo mount -t tmpfs tmpfs {tmpfs_dir}", shell=True)

    # Step 1: Create a file with the list of files to check
    with open('file_list_to_check.txt', 'w') as f:
        for file_name in file_list:
            f.write(f"{s3_bucket}/{file_prefix}{file_name}\n")

    # Step 2: List files with sizes using s5cmd
    command_list = f"s5cmd ls --include-from file_list_to_check.txt > file_list_with_sizes.txt"
    subprocess.run(command_list, shell=True)

    # Step 3: Parse the result and filter files by size
    files_to_download = []
    biopj_info_dict = {}
    with open('file_list_with_sizes.txt', 'r') as f:
        for line in f:
            parts = line.split()
            size = int(parts[0])
            file_path = parts[-1]

            if size <= max_file_size:
                files_to_download.append(file_path)
                biopj_info_dict[file_path] = BioProjectInfo(
                    file_path, os.path.join(tmp_dir, os.path.basename(file_path)), size)
            else:
                print(f"Skipping {file_path} because it is too large")

    # Write the filtered file paths to a text file
    with open('files_to_download.txt', 'w') as f:
        for file_path in files_to_download:
            f.write(f"{file_path}\n")

    # Step 4: Download the filtered files to tmp_dir
    command_cp = f"s5cmd cp -f --include-from files_to_download.txt {tmp_dir}/"
    subprocess.run(command_cp, shell=True)

    # Clean up downloaded files as they're processed
    for file_path in files_to_download:
        local_file_path = os.path.join(tmp_dir, os.path.basename(file_path))
        # Process the file (example: load the pickle file and do something)
        # with open(local_file_path, 'rb') as f:
        #     data = pickle.load(f)
        #     # Process the data
        # Remove the file after processing
        os.remove(local_file_path)

    # Clean up
    os.remove('file_list_to_check.txt')
    os.remove('file_list_with_sizes.txt')
    os.remove('files_to_download.txt')

    # Unmount tmpfs if it was mounted
    if tmp_dir == tmpfs_dir:
        subprocess.run(f"sudo umount {tmpfs_dir}", shell=True)

    return biopj_info_dict


def get_log_fold_change(true, false):
    """calculate the log fold change of true with respect to false
        if true and false is 0, then return 0
    """

    if true == 0 and false == 0:
        return 0
    elif true == 0:
        return -np.inf
    elif false == 0:
        return np.inf
    else:
        return np.log2(true / false)


def mean_diff_statistic(x, y, axis):
    """if -ve, then y is larger than x"""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def process_group(metadata_df: pd.DataFrame, group_df: pd.DataFrame) -> pd.DataFrame | None:
    """Process the given group, and return an output file
    """
    num_md_rows = metadata_df.shape[0]
    group_name = group_df['group'].iloc[0]
    # add rpm column to group_df
    group_df['rpm'] = group_df.apply(
        lambda row: row['quantifier'] / row['spots'] * NORMALIZING_CONST if row['spots'] != 0 else 0, axis=1)
    metadata_counts = {}

    for col in metadata_df.columns:
        if col == 'biosample_id':
            continue
        counts = metadata_df[col].value_counts()
        n = len(counts)
        # skip if there is only one unique value or if all values are unique
        if n == 1 or n == num_md_rows:
            continue
        metadata_counts[col] = counts

    existing_samples = list()
    target_col_runs = {}

    # iterate through the values for all columns that can be tested
    for target_col, value_counts in metadata_counts.items():
        for target_term, count in value_counts.items():
            # skip if there is only one value
            if count == 1:
                continue

            # get the biosamples that correspond to the target term
            target_term_biosamples = list(metadata_df[metadata_df[target_col] == target_term]['biosample_id'])

            # check if the same biosamples are already stored and aggregate the columns
            target_col_name = f'{target_col}\t{target_term}'

            if target_term_biosamples in existing_samples:
                existing_key = list(target_col_runs.keys())[list(target_col_runs.values()).index(target_term_biosamples)]
                target_col_name = f'{existing_key}\r{target_col_name}'
                # update the dictionary with the new name
                target_col_runs[target_col_name] = target_col_runs.pop(existing_key)
            else:
                existing_samples.append(target_term_biosamples)
                target_col_runs[target_col_name] = target_term_biosamples

    # RUN TTESTS
    output_dict = {col: [] for col in OUT_COLS}
    status = 0

    for target_col, biosamples in target_col_runs.items():
        try:
            num_true = len(biosamples)
            num_false = num_md_rows - num_true
            # get the rpm values for the target and remainng biosamples
            true_rpm = group_df[group_df['biosample_id'].isin(biosamples)]['rpm']
            false_rpm = group_df[~group_df['biosample_id'].isin(biosamples)]['rpm']
        except Exception as e:
            print(f'Error getting rpm values for {group_name} - {target_col}: {e}')
            continue

        # calculate desecriptive stats
        # NON CORRECTED VALUES
        mean_rpm_true = np.nanmean(true_rpm)
        mean_rpm_false = np.nanmean(false_rpm)
        sd_rpm_true = np.nanstd(true_rpm)
        sd_rpm_false = np.nanstd(false_rpm)

        # skip if both conditions have 0 reads
        if mean_rpm_true == mean_rpm_false == 0:
            continue

        # calculate fold change and check if any values are nan
        fold_change = get_log_fold_change(mean_rpm_true, mean_rpm_false)

        # if there are at least 4 values in each group, run a permutation test
        # otherwise run a t test
        try:
            if min(num_false, num_true) < 4:
                # scipy t test
                test_statistic, p_value = stats.ttest_ind_from_stats(mean1=mean_rpm_true, std1=sd_rpm_true, nobs1=num_true,
                                                                     mean2=mean_rpm_false, std2=sd_rpm_false, nobs2=num_false,
                                                                     equal_var=False)
            else:
                # run a permutation test
                res = stats.permutation_test((true_rpm, false_rpm), statistic=mean_diff_statistic, n_resamples=10000,
                                             vectorized=True)
                p_value = res.pvalue
                test_statistic = res.statistic
        except Exception as e:
            print(f'Error running statistical test for {group_name} - {target_col}: {e}')
            continue

        # extract metadata_field (column names) and metadata_value (column values)
        metadata_tmp = target_col.split('\r')  # pairs of metadata_field and metadata_value for aggregated columns
        metadata_field = '\t'.join(pair.split('\t')[0] for pair in metadata_tmp)
        metadata_value = '\t'.join(pair.split('\t')[1] for pair in metadata_tmp)

        # add values to output dict
        output_cols = (status, metadata_field, metadata_value, num_true, num_false, mean_rpm_true, mean_rpm_false,
                       sd_rpm_true, sd_rpm_false, fold_change, test_statistic, p_value)
        for i, col in enumerate(OUT_COLS):
            output_dict[col].append(output_cols[i])

    # create output dataframe
    output_df = pd.DataFrame(output_dict)
    output_df.sort_values(by='p_value')
    return output_df


def process_bioproject(bioproject: BioProjectInfo, main_df: pd.DataFrame) -> pd.DataFrame | None:
    """Process the given bioproject, and return an output file - concatenation of several group outputs
    """
    bioproject.load_metadata()  # access this df as bioproject.metadata_df

    if bioproject.metadata_df is None or bioproject.metadata_row_count <= 2:
        print(f"Skipping {bioproject.name} since its metadata is empty or too small")
        return None

    # get subset of main_df that belongs to this bioproject
    subset_df = main_df[main_df['bio_project'] == bioproject.name] & main_df['biosample'].isin(bioproject.metadata_df['biosample'])

    # group processing
    group_to_df_map = {group: subset_df[subset_df['group'] == group] for group in subset_df['group'].unique()}
    group_output_dfs = []
    for group in group_to_df_map:
        output_file = process_group(bioproject.metadata_df, group_to_df_map[group])
        group_output_dfs.append(output_file)

    bioproject.delete_metadata()

    # remove part of main_df that belongs to this bioproject to free up some space?
    # use subset_df to speed up the process
    # main_df.drop(subset_df.index)

    # concatenate all the group output files into one
    return pd.concat(group_output_dfs)


def run_on_file(data_file: pd.DataFrame, input_info: tuple[str, str]) -> None:
    """Run MWAS on the given data file
    """
    # ================
    # PRE-PROCESSING
    # ================

    # TODO: HASHING THE INPUT FILE
    # hash_code = hash_file(data_file)
    # check if dir <hash_code> in s3
    # if yes, get its output csv
    # otherwise, create the dir and store the input_df there as a csv

    # CREATE MAIN DATAFRAME
    runs = data_file['run'].unique()
    main_df = get_bioprojects_df(runs)
    if main_df is None:
        return
    # merge the main_df with the input_df (must assume both have run column)
    main_df = main_df.merge(data_file, on='run', how='inner')  # TODO: account for missing samples in user input - fill in with MAP_UNKNOWN
    del data_file, runs
    main_df.groupby('bio_project')

    # TODO: STORE THE MAIN DATAFRAME IN S3
    # temporarily rename the columns to standard names before storing to s3
    # store the main_df in the dir <hash_code> in s3
    # restore the column names back to being general names

    # BLOCK INFO PREPARATION
    bioprojects_list = main_df['bio_project'].unique()
    num_bioprojects = len(bioprojects_list)
    num_blocks = max(1, num_bioprojects // BLOCK_SIZE)

    # ================
    # PROCESSING
    # ================

    for i in range(0, num_blocks):  # for each block in bioprojects
        # GET LIST OF BIOPROJECTS IN THIS CURRENT BLOCK
        if (i + 1) * BLOCK_SIZE > num_bioprojects:
            biopj_block = bioprojects_list[i * BLOCK_SIZE:]
        else:
            biopj_block = bioprojects_list[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE]

        # GET METADATA FOR BIOPROJECTS IN THIS BLOCK
        biopj_info_dict = metadata_retrieval(biopj_block)

        if not SCHEDULING:
            random.shuffle(biopj_block)
            # PROCESS THE BLOCK
            for biopj in biopj_info_dict.keys():
                output_file = process_bioproject(biopj_info_dict[biopj], main_df)

                # STORE THE OUTPUT FILE IN temp folder on disk as a file <biopj>_output.csv
                with open(f"{OUTPUT_DIR_DISK}/{biopj}_output.csv", 'w') as f:
                    output_file.to_csv(f)

        else:  # SCHEDULING
            # TODO: IMPLEMENT SCHEDULING
            raise NotImplementedError

        # now we're done processing an entire block
        # TODO: STORE THE OUTPUT FILES IN S3
        # concatenate all the output files in the block into one csv (spans multiple bioprojects)
        # and then append-write to output file in the folder <hash_code> (create the output file if it doesn't exist yet)

        # FREE UP EVERY ROW IN THE MAIN DATAFRAME THAT BELONGS TO THE CURRENT BLOCK BY INDEXING VIA BIOPROJECT
        main_df = main_df[~main_df['bio_project'].isin(biopj_block)]

    # ================
    # POST-PROCESSING
    # ================


if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # FIX THIS LATER

    # Check if the correct number of arguments is provided
    arg1 = sys.argv[1]
    if len(sys.argv) < 1 or arg1 in ('-h', '--help'):
        print("Usage:")
        print("python mwas_general.py data_file.csv")
        sys.exit(1)
    elif arg1.endswith('.csv'):
        try:  # reading the input file
            input_df = pd.read_csv(arg1)  # arg1 is a file path
            # rename group and quantifier columns to standard names, and also save original names
            group_by, quantifying_by = input_df.columns[1], input_df.columns[2]
            input_df.rename(columns={input_df.columns[1]: 'group', input_df.columns[2]: 'quantifier'}, inplace=True)

            # assume it has three columns: run, group, quantifier. And the group and quantifier columns have special names
            if 'run' not in input_df.columns and len(input_df.columns) != 3:
                print("Data file must have three columns in this order: run, <group>, <quantifier>")
                sys.exit(1)
            # check if run and group contain string values and quantifier contains numeric values
            if (input_df['run'].dtype != 'object' or input_df['group'].dtype != 'object'
                    or input_df['quantifier'].dtype not in ('float64', 'int64')):
                print("Run column must contain string values")
                sys.exit(1)
        except FileNotFoundError:
            print("File not found")
            sys.exit(1)

        run_on_file(input_df, (group_by, quantifying_by))
    else:
        print("Invalid arguments")
        sys.exit(1)
