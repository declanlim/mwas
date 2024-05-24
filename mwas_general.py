"""Generalized MWAS
"""
# built-in libraries
import os
import sys
import platform
import subprocess
import pickle
# import warnings
# import logging
import random

# import required libraries
import psycopg2
import pandas as pd
import numpy as np
import scipy.stats as stats

# sql constants
CONNECTION_INFO = {
    'host': 'serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com',
    'database': 'summary',
    'user': 'public_reader',
    'password': 'serratus'
}
query = """
    SELECT bio_project, bio_sample, run, spots
    FROM %s -- srarun
    WHERE run in (%s)
"""

# system & path constants
PICKLE_DIR = 'pickles'  # will be relative to working directory
OUTPUT_DIR_DISK = 'outputs'
S3_METADATA_DIR = 's3://serratus-biosamples/mwas_setup/bioprojects'
# S3_OUTPUT_DIR = 's3://serratus-biosamples/mwas_outputs'
OS = platform.system()
SHELL_PREFIX = 'wsl ' if OS == 'Windows' else ''
DEFAULT_MOUNT_POINT = '/mnt/mwas'
TMPFS_DIR = '/mnt/tmpfs'
SYSTEM_MOUNTS = 'wmic logicaldisk get name' if OS == 'Windows' else 'mount'

# processing constants
BLOCK_SIZE = 1000
MAX_PROJECT_SIZE = 50 * 1024 ** 2  # 50 MB
PICKLE_SPACE_LIMIT = 2 * 1024 ** 3  # 2 GB
SCHEDULING = False

# special constants
BLACKLIST = set()  # list of bioprojects that are too large

# stats constants
MAP_UNKNOWN = 0  # maybe np.nan
NORMALIZING_CONST = 1000000  # 1 million
OUT_COLS = ['status', 'metadata_field', 'metadata_value', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
            'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value']


class BioProjectInfo:
    """Class to store information about a bioproject"""

    def __init__(self, name: str, system_metadata_file_path: str, metadata_size: int = None) -> None:
        self.name = name
        self.metadata_path = system_metadata_file_path
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


class MountTmpfs:
    """A mounted tmpfs filesystem referencer
    """

    def __init__(self, mount_point: str = DEFAULT_MOUNT_POINT, alloc_space: int = None) -> None:
        self.mount_point = mount_point
        self.alloc_space = alloc_space  # if None, then it's a dynamic mount
        self.space_used = 0  # bytes of actual space currently in use
        self.is_mounted = False

    def is_tmpfs_mounted(self) -> bool:
        """Checks if the given mount point is mounted as tmpfs or exists as a logical disk in Windows."""
        if OS == 'Windows':
            return False  # Windows does not support tmpfs mounting
        else:
            return self.is_tmpfs_mounted_unix()

    def is_tmpfs_mounted_unix(self) -> bool:
        """Check if the given mount point is mounted as tmpfs in Unix."""
        try:
            mounts = subprocess.check_output('mount', shell=True).decode().split('\n')
            for mount in mounts:
                parts = mount.split()
                if len(parts) > 4 and parts[2] == self.mount_point and parts[4] == 'tmpfs':
                    return True
            return False
        except subprocess.CalledProcessError as e:
            print(f"Failed to check mounted drives in Unix: {e}")
            return False

    def mount(self) -> None:
        """Mounts the tmpfs filesystem"""
        if OS == 'Windows':
            if not os.path.exists(PICKLE_DIR):
                os.mkdir(PICKLE_DIR)
            self.mount_point = None
            print("Windows does not support tmpfs mounting. Resorting to using working dir folder without mount...")
            return
        if not self.is_tmpfs_mounted():
            print(f"{self.mount_point} was not mounted as tmpfs (RAM), so mounting it now.")
            if OS == 'Windows':
                # Windows-specific mounting
                pass
            else:
                # Unix-specific mounting
                alloc_space = f"-o size={self.alloc_space}" if self.alloc_space else ""
                subprocess.run(
                    f"sudo mount {alloc_space} -t tmpfs {self.mount_point.split()[-1]} {self.mount_point}",
                    shell=True
                )
            self.is_mounted = True
        else:
            print(f"{self.mount_point} is already mounted as type tmpfs")

        self.is_mounted = True

    def unmount(self) -> None:
        """Unmounts the tmpfs filesystem"""
        if self.is_tmpfs_mounted():
            if OS == 'Windows':
                # Windows-specific file setup
                os.rmdir(PICKLE_DIR)
                pass
            else:
                # Unix-specific unmounting
                subprocess.run(f"sudo umount -f {self.mount_point}", shell=True)
                self.is_mounted = False
                print(f"{self.mount_point} was unmounted")
        else:
            print(f"{self.mount_point} is not mounted as tmpfs")


def metadata_retrieval(biopj_block: list[str], storage: MountTmpfs) -> dict[str, BioProjectInfo]:
    """Retrieve metadata for the given bioprojects, and organize them in a dictionary
    storage will usually be the tmpfs mount point mount_tmpfs
    """
    ls_file_name = "ls_batch_command_list.txt"
    size_list_file = "file_list_with_sizes.txt"
    cp_file_name = "cp_batch_command_list.txt"

    # handling windows os special treatment
    file_storage = storage.mount_point
    if storage.mount_point is None:
        # this implies we're using the working directory. Make sure PICKLE_DIR exists
        if os.path.exists(PICKLE_DIR):
            file_storage = PICKLE_DIR
        else:
            print(f"Error: {PICKLE_DIR} does not exist. Exiting.")
            sys.exit(1)
    elif not os.path.ismount(file_storage):
        print(f"Error: {file_storage} is not mounted. Exiting.")
        sys.exit(1)

    # Create a file with the list of files to check (although this takes an extra loop, s5cmd makes it worth it)
    with open(ls_file_name, 'w+') as f:
        # writing a bucket path for each line in our file list file. A file format is needed for s5cmd
        for biopj in biopj_block:
            f.write(f'ls {S3_METADATA_DIR}/{biopj}\n')

    # List files with sizes using s5cmd (remember, this only looks through a single block of bioprojects)
    # note: s5cmd does not support piping, so we have to use awk here as opposed to in the previous step for each line
    command_list = SHELL_PREFIX + f"s5cmd run {ls_file_name} | " + SHELL_PREFIX + f"awk \'{{print $(NF-1), $NF}}\' > {size_list_file}"
    subprocess.run(command_list, shell=True)
    os.remove(ls_file_name)  # useless now that we have the file_list_with_sizes.txt

    # filter files by size, and also set create each BioProjectInfo object and load them into a dictionary
    biopj_info_dict = {}
    total_size = 0
    with open(size_list_file, 'r') as read_file, open(cp_file_name, 'w') as write_file:
        for line in read_file:
            parts = line.split()
            # remember, we're reading from a ls output file that was reformatted by awk: parts[0] is an int, parts[1] is <biopj>.pickle
            size, file = int(parts[0]), parts[1]

            biopj_name = file.split('.')[0]  # get <biopj> from <biopj>.pickle
            if size <= MAX_PROJECT_SIZE and biopj_name not in BLACKLIST:
                write_file.write(f"cp -f {S3_METADATA_DIR}/{file} {file_storage}\n")
                biopj_info_dict[biopj_name] = BioProjectInfo(biopj_name, f"{file_storage}/{file}", size)
                total_size += size
            else:
                print(f"Skipping {biopj_name} because it is too large, with {size} bytes.")
            if total_size >= PICKLE_SPACE_LIMIT:
                print(f"Reached the limit of {PICKLE_SPACE_LIMIT} bytes. Only downloaded {len(biopj_info_dict)} files.")
                break
    os.remove(size_list_file)

    # Download the filtered files to tmp_dir
    command_cp = f"s5cmd run {cp_file_name}"
    subprocess.run(SHELL_PREFIX + command_cp, shell=True)
    os.remove(cp_file_name)

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
            biosample_ids = np.array(group_df['bio_sample'].tolist())

            rpm = group_df.apply(
                lambda row: row['quantifier'] / row['spots'] * NORMALIZING_CONST if row['spots'] != 0 else 0, axis=1)

            mask = np.isin(biosample_ids, biosamples)
            true_rpm = rpm[mask]
            false_rpm = rpm[~mask]
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
    subset_df = main_df[main_df['bio_project'] == bioproject.name]
    subset_df = subset_df[subset_df['bio_sample'].isin(bioproject.metadata_df['biosample_id'])]

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


def run_on_file(data_file: pd.DataFrame, input_info: tuple[str, str], storage: MountTmpfs) -> None:
    """Run MWAS on the given data file
    input_info is a tuple of the group and quantifier column names
    storage will usually be the tmpfs mount point mount_tmpfs
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
    main_df = main_df.merge(data_file, on='run', how='outer')
    main_df.fillna(MAP_UNKNOWN, inplace=True)
    del data_file, runs
    main_df.groupby('bio_project')

    # TODO: STORE THE MAIN DATAFRAME IN S3
    # temporarily rename the columns to standard names before storing to s3
    # store the main_df in the dir <hash_code> in s3
    # restore the column names back to being general names

    # BLOCK INFO PREPARATION
    bioprojects_list = main_df['bio_project'].unique()
    random.shuffle(bioprojects_list)
    num_bioprojects = len(bioprojects_list)
    num_blocks = max(1, num_bioprojects // BLOCK_SIZE)

    # ================
    # PROCESSING
    # ================

    i = 0
    while i < num_bioprojects:  # for each block (roughly) in bioprojects
        # GET LIST OF BIOPROJECTS IN THIS CURRENT BLOCK
        if i + BLOCK_SIZE > num_bioprojects:  # currently inside a non-full block
            biopj_block = bioprojects_list[i:]
        else:
            biopj_block = bioprojects_list[i: i + BLOCK_SIZE]

        # GET METADATA FOR BIOPROJECTS IN THIS BLOCK
        biopj_info_dict = metadata_retrieval(biopj_block, storage)
        if len(biopj_info_dict) < BLOCK_SIZE:
            # handling since we reached the space limit before downloading a full blocks worth of bioprojects
            i += len(biopj_info_dict)
        else:
            i += BLOCK_SIZE  # move to the next block

        del biopj_block

        if not SCHEDULING:
            # random.shuffle(biopj_block) is this even necessary since it's iterating over a dict?
            # PROCESS THE BLOCK
            for biopj in biopj_info_dict.keys():
                output_file = process_bioproject(biopj_info_dict[biopj], main_df)

                # OUTPUT FILE (FOR THIS PARTICULAR BIOPROJECT)
                if output_file is not None and output_file.shape[0] > 1:
                    # STORE THE OUTPUT FILE IN temp folder on disk as a file <biopj>_output.csv
                    with open(f"{OUTPUT_DIR_DISK}/{biopj}_output.csv", 'w') as f:
                        output_file.to_csv(f, index=False)
                elif output_file is not None and output_file.shape[0] == 1:
                    print(f"Output file for {biopj} is empty. Not creating a file for it.")
                else:
                    print(f"There was a problem with making an output file for {biopj}")

        else:  # SCHEDULING
            # TODO: IMPLEMENT SCHEDULING
            raise NotImplementedError

        # now we're done processing an entire block
        # TODO: STORE THE OUTPUT FILES IN S3
        # concatenate all the output files in the block into one csv (spans multiple bioprojects)
        # and then append-write to output file in the folder <hash_code> (create the output file if it doesn't exist yet)

        # FREE UP EVERY ROW IN THE MAIN DATAFRAME THAT BELONGS TO THE CURRENT BLOCK BY INDEXING VIA BIOPROJECT
        main_df = main_df[~main_df['bio_project'].isin(biopj_info_dict.keys())]

    # ================
    # POST-PROCESSING
    # ================


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')  # FIX THIS LATER

    # Check if the correct number of arguments is provided
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python mwas_general.py data_file.csv")
        sys.exit(1)
    elif sys.argv[1].endswith('.csv'):
        try:  # reading the input file
            input_df = pd.read_csv(sys.argv[1])  # arg1 is a file path
            # rename group and quantifier columns to standard names, and also save original names
            group_by, quantifying_by = input_df.columns[1], input_df.columns[2]
            input_df.rename(columns={
                input_df.columns[0]: 'run', input_df.columns[1]: 'group', input_df.columns[2]: 'quantifier'
            }, inplace=True)

            # assume it has three columns: run, group, quantifier. And the group and quantifier columns have special names
            if len(input_df.columns) != 3:
                print("Data file must have three columns in this order: <run>, <group>, <quantifier>")
                sys.exit(1)
            # check if run and group contain string values and quantifier contains numeric values
            if (input_df['run'].dtype != 'object' or input_df['group'].dtype != 'object'
                    or input_df['quantifier'].dtype not in ('float64', 'int64')):
                print("run and group column must contain string values, and quantifier column must contain numeric values")
                sys.exit(1)
        except FileNotFoundError:
            print("File not found")
            sys.exit(1)

        # MOUNT TMPFS
        mount_tmpfs = MountTmpfs()
        mount_tmpfs.mount()
        if not mount_tmpfs.is_mounted:
            print("Did not mount tmpfs, likely because this is a Windows system. Using working directory instead.")
        # CREATE OUTPUT DIRECTORY
        if not os.path.exists(OUTPUT_DIR_DISK):
            os.mkdir(OUTPUT_DIR_DISK)

        # RUN MWAS
        run_on_file(input_df, (group_by, quantifying_by), mount_tmpfs)

        # UNMOUNT TMPFS
        mount_tmpfs.unmount()
        sys.exit(0)
    else:
        print("Invalid arguments")
        sys.exit(1)
