"""Generalized MWAS
"""
# built-in libraries
import math
import os
import sys
import platform
import subprocess
import pickle
import time
import logging
import json
from atexit import register
from shutil import rmtree
from typing import Any
from multiprocessing import Manager, Pool, cpu_count

import boto3
# import required libraries
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import permutation_test, ttest_ind_from_stats

# sql constants
SERRATUS_CONNECTION_INFO = {
    'host': 'serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com',
    'database': 'summary',
    'user': 'public_reader',
    'password': 'serratus'
}
LOGAN_CONNECTION_INFO = {
    'host': 'serratus-aurora-20210406.cluster-ro-ccz9y6yshbls.us-east-1.rds.amazonaws.com',
    'database': 'logan',
    'user': 'public_reader',
    'password': 'serratus'
}
META_QUERY = """
    SELECT * FROM ethan_mwas_20240717
    WHERE bioproject in (%s)
"""
LOGAN_SRA_QUERY = ("""
    SELECT bioproject as bio_project, biosample as bio_sample, acc as run, ((CAST(mbases AS BIGINT) * 1000000) / avgspotlen) AS spots 
    FROM sra
    WHERE acc in (%s)
    """, """
    SELECT bioproject as bio_project, biosample as bio_sample, acc as run
    FROM sra
    WHERE acc in (%s)
""")

DECODING_ERRORS = {
    12: 'Too large to process.',
    11: 'Blacklisted.',
    10: 'placeholder file for unavailable bioproject (i.e. Dupe-Bug).',
    9: 'FAILED.',
    8: 'Other error.',
    7: 'CSV reading error.',
    6: 'All rows were scrambled.',
    5: 'Found invalid biosample_id rows.',
    4: 'Originally empty in raw file.',
    3: 'Less than 4 biosamples.',
    2: 'No sets were generated.',
    1: 'Empty file.',
    0: 'No issues.'
}


def decode_comment_code(comment_code: int) -> str:
    """Converts a code to a comment."""
    comments = []
    for bit, msg in DECODING_ERRORS.items():
        if comment_code & (1 << bit):
            comments.append(msg)
    return ' '.join(comments)


# path constants
S3_METADATA_DIR = 's3://serratus-biosamples/condensed-bioproject-metadata'
S3_OUTPUT_DIR = 's3://serratus-biosamples/mwas_data/'
TEMP_LOCAL_BUCKET = None  # tbd when hash code is provided
PROBLEMATIC_BIOPJS_FILE = 'bioprojects_ignored.txt'
PREPROC_LOG_FILE = 'preprocessing_log.txt'
PROGRESS_FILE = 'progress.json'
OUTPUT_CSV_FILE = 'mwas_output.csv'
OUTPUT_FILES_DIR = 'outputs'
PROC_LOG_FILES_DIR = 'proc_logs'

# system constants
OS = platform.system()
SHELL_PREFIX = 'wsl ' if OS == 'Windows' else ''

# special constants
BLACKLIST = {"PRJEB37886", "PRJNA514245", "PRJNA716984", "PRJNA731148", "PRJNA631508", "PRJNA665224", "PRJNA716985", "PRJNA479871", "PRJNA715749", "PRJEB11419",
             "PRJNA750736", "PRJNA525951", "PRJNA720050", "PRJNA731152", "PRJNA230403", "PRJNA675921", "PRJNA608064", "PRJNA486548",
             "nan", "PRJEB43828", "PRJNA609094", "PRJNA686984", "PRJNA647773", "PRJNA995950"
             }  # list of bioprojects that are too large
BLACKLIST_2 = {"PRJNA614995", "PRJNA704697", "PRJNA738870", "PRJNA731149", "PRJNA248792", "PRJNA218110", "PRJEB40277", "PRJNA808151", "PRJDB17001", "PRJNA612578",
               "PRJNA767338", "PRJDB11811", "PRJEB2136", "PRJNA793773", "PRJNA983279", "PRJNA630714", "PRJNA242847", "PRJEB47340", "PRJEB2141", "PRJEB14362", "PRJNA738869",
               "PRJEB3084", "PRJNA743046", "PRJNA689853", "PRJNA186035", "PRJNA315192"
               }
BLACKLIST = BLACKLIST.union(BLACKLIST_2)
BLACKLISTED_METADATA_FIELDS = {
    'publication_date', 'center_name', 'first_public', 'last_public', 'last_update', 'INSDC center name', 'INSDC first public',
    'INSDC last public', 'INSDC last update', 'ENA center name', 'ENA first public', 'ENA last public', 'ENA last update',
    'ENA-FIRST-PUBLIC', 'ENA-LAST-UPDATE', 'DDBJ center name', 'DDBJ first public', 'DDBJ last public', 'DDBJ last update',
    'Contacts/Contact/Name/First', 'Contacts/Contact/Name/Middle', 'Contacts/Contact/Name/Last', 'Contacts/Contact/@email',
    'Name/@url', 'name/@url', 'collected_by', 'when', 'submission_date'
}

# constants
MAP_UNKNOWN = 0  # maybe np.nan. Use -1 for when IMPLOICIT_ZEROS is False
NORMALIZING_CONST = 1000000  # 1 million
LAMBDA_RANGE = (128 * 1024 ** 2, 2048 * 1024 ** 2)  # 128 MB to 10240 MB (max lambda memory size
PERCENT_USED = 0.8  # 80% of the memory will be used
MAX_RAM_SIZE = PERCENT_USED * LAMBDA_RANGE[1]
LAMBDA_CPU_CORES = 6
PROCESS_OVERHEAD_BOUND = 84 * 1024 ** 2  # 84 MB (run module_size_test to determine this, then round up a bit)
OUT_COLS = [
    'bioproject', 'group', 'metadata_field', 'metadata_value', 'test', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
    'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value', 'true_biosamples', 'false_biosamples', 'test_time_duration'
]

# logging
use_logger = True
logging_level = 2  # 0: no logging, 1: minimal logging, 2: verbose logging
logging.basicConfig(level=logging.INFO)
logger = None

# flags
IMPLICIT_ZEROS = True  # TODO: implement this flag for when it's False
GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = 3  # if group has at least this many nonzeros, then it's okay. Note, this flag is only used if IMPLICIT_ZEROS is True
ALREADY_NORMALIZED = False  # if it's false, then we need to normalize the data by dividing quantifier by spots
P_VALUE_THRESHOLD = 0.005
INCLUDE_SKIPPED_GROUP_STATS = False  # if True, then the output will include the stats of the skipped tests
TEST_BLACKLISTED_METADATA_FIELDS = False  # if False, then the metadata fields in BLACKLISTED_METADATA_FIELDS will be ignored


def log_print(msg: Any, lvl: int = 1) -> None:
    """Print a message if the logging level is appropriate"""
    global logging_level, use_logger
    if lvl <= logging_level:
        if use_logger and isinstance(logger, logging.Logger):
            logger.info(msg)
        else:
            print(msg)


def lambda_handler(event: dict, context: Any) -> dict:
    """Lambda handler for MWAS"""
    # get the data from event
    bioproject_info = event['bioproject_info']
    main_df_link = event['main_df_link']
    job_window = event['job_window']
    id = event['id']
    FLAGS = event['FLAGS']
    # update flags
    for flag in FLAGS:
        globals()[flag] = FLAGS[flag]

    # create logger
    global logger
    logger = logging.getLogger(f"log_{bioproject_info['name']}_job{id}.txt")
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(f"log_{bioproject_info['name']}_job{id}.txt")
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # get the main_df
    try:
        command_cp = f"s5cmd cp -f {main_df_link} main_df.pickle"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        with open('main_df.pickle', 'rb') as f:
            main_df = pickle.load(f)
        os.remove('main_df.pickle')

    except Exception as e:
        log_print(f"Error in loading main_df: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in loading main_df: {e}"}

    # get the bioproject info
    bioproject = BioProjectInfo(**bioproject_info)
    bioproject.retrieve_metadata()
    result = bioproject.process_bioproject(main_df, job_window, id)
    if result is None:
        return {'statusCode': 500, 'body': f"Error in processing bioproject {bioproject.name} job {id}"}
    # make result file
    with open('result.txt', 'w') as f:
        f.write(result)

    # upload the result and log files to s3
    try:
        command_cp = f"s5cmd cp -f result.txt {S3_OUTPUT_DIR}/{OUTPUT_FILES_DIR}/result_{bioproject.name}_job{id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove('result.txt')

        command_cp = f"s5cmd cp -f log_{bioproject.name}_job{id}.txt {S3_OUTPUT_DIR}/{PROC_LOG_FILES_DIR}/log_{bioproject.name}_job{id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove(f"log_{bioproject.name}_job{id}.txt")

    except Exception as e:
        log_print(f"Error in syncing output: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in putting output to s3: {e}"}

    return {'statusCode': 200, 'body': f'MWAS processing completed for {bioproject.name} job {id}'}


class BioProjectInfo:
    """Class to store information about a bioproject"""

    def __init__(self, name: str, metadata_file_size: int, n_biosamples: int,
                 n_sets: int, n_permutation_sets: int, n_skippable_permutation_sets: int,
                 n_groups: int, n_skipped_groups: int, num_lambda_jobs: int = 0, num_conc_procs: int = 0,
                 groups: list[str] = None, jobs: list[tuple[dict[str, tuple[int, int]], int]] = None) -> None:
        self.name = name
        self.md_file_size = metadata_file_size
        self.n_sets = n_sets
        self.n_permutation_sets = n_permutation_sets
        self.n_skippable_permutation_sets = n_skippable_permutation_sets
        self.n_actual_permutation_sets = n_permutation_sets - n_skippable_permutation_sets \
            if not TEST_BLACKLISTED_METADATA_FIELDS else n_permutation_sets
        self.n_biosamples = n_biosamples
        self.n_groups = n_groups
        self.n_skipped_groups = n_skipped_groups
        self.metadata_df = None
        self.metadata_ref_lst = None
        self.metadata_path = None
        self.rpm_map = None
        self.num_lambda_jobs = num_lambda_jobs
        self.num_conc_procs = num_conc_procs
        self.groups = [] if groups is None else groups
        self.jobs = [] if jobs is None else jobs

    def to_dict(self) -> dict:
        """Convert the bioproject info to a dictionary"""
        return {
            'name': self.name,
            'metadata_file_size': self.md_file_size,
            'n_biosamples': self.n_biosamples,
            'n_sets': self.n_sets,
            'n_permutation_sets': self.n_permutation_sets,
            'n_skippable_permutation_sets': self.n_skippable_permutation_sets,
            'n_groups': self.n_groups,
            'n_skipped_groups': self.n_skipped_groups,
            'num_lambda_jobs': self.num_lambda_jobs,
            'num_conc_procs': self.num_conc_procs,
            'groups': self.groups,
            'jobs': self.jobs
        }

    def batch_lambda_jobs(self, main_df: pd.DataFrame, time_constraint: int) -> tuple[int, int]:
        """
        time_constraint: in seconds
        """
        # choosing lambda size
        mem_width = self.n_biosamples * 240000 + PROCESS_OVERHEAD_BOUND
        n_conc_procs = LAMBDA_CPU_CORES
        while mem_width * n_conc_procs > MAX_RAM_SIZE:
            n_conc_procs -= 1
        if n_conc_procs == 0:
            log_print(f"Error: not enough memory to run a single test on a lambda functions for bio_project {self.name}")
            return 0, 0
        self.num_conc_procs = n_conc_procs

        time_per_test = self.n_biosamples * 0.0003
        total_tests = (self.n_groups - self.n_skipped_groups) * self.n_actual_permutation_sets
        lambda_area = (math.floor(time_constraint / time_per_test) * n_conc_procs)
        self.num_lambda_jobs = math.ceil(total_tests / lambda_area)

        self.jobs = []
        if self.num_lambda_jobs > 1:

            subset_df = main_df[main_df['bio_project'] == self.name]
            group_counts = subset_df['group'].value_counts()
            self.groups = group_counts[group_counts >= GROUP_NONZEROS_ACCEPTANCE_THRESHOLD].index.tolist()

            num_tests_added = curr_group = left_off_at = 0
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
                    if num_tests_added % self.n_actual_permutation_sets == 0:  # finished adding tests for a group
                        focus_groups[self.groups[curr_group]] = (started_at, left_off_at)
                        curr_group += 1
                        started_at = left_off_at = 0
                if left_off_at > 0:
                    focus_groups[self.groups[curr_group]] = (started_at, left_off_at)
                self.jobs.append((focus_groups, lambda_size))
        elif self.num_lambda_jobs == 1:
            self.jobs.append('full')

        assert self.num_lambda_jobs == len(self.jobs)
        return self.n_actual_permutation_sets, self.num_lambda_jobs

    def dispatch_all_lambda_jobs(self, link: str, lam_client: boto3.client) -> None:
        """lam_client: lambda_client = boto3.client('lambda')

        note lambda handler will call self.process_bioproject(...)
        """
        bioproject_info = self.to_dict()

        flags = {
            'IMPLICIT_ZEROS': int(IMPLICIT_ZEROS),
            'GROUP_NONZEROS_ACCEPTANCE_THRESHOLD': GROUP_NONZEROS_ACCEPTANCE_THRESHOLD,
            'ALREADY_NORMALIZED': int(ALREADY_NORMALIZED),
            'P_VALUE_THRESHOLD': P_VALUE_THRESHOLD,
            'INCLUDE_SKIPPED_GROUP_STATS': int(INCLUDE_SKIPPED_GROUP_STATS),
            'TEST_BLACKLISTED_METADATA_FIELDS': int(TEST_BLACKLISTED_METADATA_FIELDS),
            'MAP_UNKNOWN': MAP_UNKNOWN
        }

        log_print(f"dispatching all {self.num_lambda_jobs} lambda jobs for {self.name}")
        # non permutation test lambda(TODO: lambda[s]?)
        lam_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps({
                'bioproject_info': bioproject_info,
                'main_df_link': link,
                'job_window': 'full',  # empty implies all groups
                'id': 0,  # 0 implies this is the non-perm tests lambda
                'flags': flags
            })
        )

        for i, job in enumerate(self.jobs):
            log_print(f"dispatching lambda job {i + 1} for {self.name}")
            # Invoke the Lambda function
            lam_client.invoke(
                FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps({
                    'bioproject_info': bioproject_info,
                    'main_df_link': link,
                    'job_window': job,
                    'id': i + 1,
                    'flags': flags
                })
            )

    def retrieve_metadata(self) -> None:
        """load metadata pickle file from s3 into memory as a DataFrame and ref list.
        Note: we should only use this if we're currently working with the bioproject
        """
        try:
            command_cp = f"s5cmd cp -f {S3_METADATA_DIR}/{self.name}.mwaspkl temp_file.pickle"
            subprocess.run(SHELL_PREFIX + command_cp, shell=True)
            with open('temp_file.pickle', 'rb') as f:
                self.metadata_ref_lst = pickle.load(f)
                self.metadata_df = pickle.load(f)
                if len(self.metadata_ref_lst) != len(self.metadata_df):
                    log_print("Warning: metadata ref list and metadata df are not the same length. Requires updating mwas metadata table")
                self.n_biosamples = len(self.metadata_ref_lst)
            os.remove('temp_file.pickle')
        except Exception as e:
            log_print(f"Error in loading metadata for {self.name}: {e}", 2)

    def build_rpm_map(self, input_df: pd.DataFrame, groups_focus: dict | None) -> None:
        """Build the rpm map for this bioproject
        """
        if self.rpm_map is not None:  # if the rpm map has already been built
            return
        # get subset of main_df that belongs to this bioproject
        # rpm map building
        time_build = time.time()
        groups = input_df['group'].unique()
        global MAP_UNKNOWN
        if not IMPLICIT_ZEROS:
            MAP_UNKNOWN = -1  # to allow for user provided 0,
            # we must use a negative number to indicate unknown (user should never provide a negative number)

        # groups_rpm_map is a dictionary with keys as groups and values as a list of two items:
        # the rpm list for the group and a boolean - True => skip the group
        groups_rpm_map = {group: [np.full(self.n_biosamples, MAP_UNKNOWN, float), False] for group in
                          groups if groups in groups_focus or groups_focus is None}
        # remember, numpy arrays work better with preallocation

        missing_biosamples = set()
        for g in groups:
            group_subset = input_df[input_df['group'] == g]
            if IMPLICIT_ZEROS:
                if GROUP_NONZEROS_ACCEPTANCE_THRESHOLD > 0:
                    num_provided = group_subset['quantifier'].count()
                    if num_provided < GROUP_NONZEROS_ACCEPTANCE_THRESHOLD:
                        groups_rpm_map[g][1] = True
            else:
                num_provided = group_subset['quantifier'].count()
                if num_provided < 4:  # because you need at least 4 values to run a test
                    groups_rpm_map[g][1] = True

            for biosample in group_subset['bio_sample'].unique():
                if biosample in missing_biosamples:
                    continue
                try:
                    i = self.metadata_ref_lst.index(biosample)  # get the index of the biosample in the reference list
                except ValueError:
                    missing_biosamples.add(biosample)
                    continue

                biosample_subset = group_subset[group_subset['bio_sample'] == biosample]
                reads, spots = biosample_subset['quantifier'], biosample_subset['spots']

                # get mean of the quantifier for this biosample in this group and load that into the rpm map
                if len(biosample_subset) > 1:
                    if ALREADY_NORMALIZED:
                        groups_rpm_map[g][0][i] = np.mean(reads.values)
                    else:
                        groups_rpm_map[g][0][i] = np.mean([reads.values[x] / (spots.values[x] * NORMALIZING_CONST)
                                                           if spots.values[x] != 0 else MAP_UNKNOWN
                                                           for x in range(len(biosample_subset))])
                else:
                    if ALREADY_NORMALIZED:
                        groups_rpm_map[g][0][i] = reads.values[0]
                    else:
                        groups_rpm_map[g][0][i] = reads.values[0] / spots.values[0] * NORMALIZING_CONST \
                            if spots.values[0] != 0 else MAP_UNKNOWN

        del groups
        if missing_biosamples:  # this is an issue due to the raw csvs not having retrieved certain metadata from ncbi,
            # possibly because the metadata is outdated. Therefore, if this is found, the raw csvs must be remade and then condensed again
            log_print(
                f"Not found in ref lst: {', '.join(missing_biosamples)} for {self.name} - "
                f"this implies there was no mention of this biosample in the bioproject's metadata file, despite the biosample having "
                f"been provided by the user. Though, this doesn't necessarily mean it's there's no metadata for the biosample on NCBI",
                2)
        log_print(f"Built rpm map for {self.name} in {round(time.time() - time_build, 2)} seconds\n")

    def process_bioproject(self, subset_df: pd.DataFrame, id: int, job: tuple[dict[str, tuple[int, int]], int]) -> str | None:
        """Process the given bioproject, and return an output file - concatenation of several group outputs
        """
        time_start = time.time()
        identifier = f"B{id}/{self.num_lambda_jobs}"
        log_print(f"RETRIEVING METADATA FOR {self.name}...")
        self.retrieve_metadata()
        if self.metadata_df.shape[0] < 4:
            # although no metadata should have 0 rows, since those would be condensed to an empty file
            # and then filtered out in metadata_retrieval, this is just a safety check
            log_print(f"Skipping {self.name} since its metadata is empty or too small. "
                      f"(Note: if you see this message, there is a discrepancy in the metadata files)")
            return None
        log_print(f"BUILDING RPM MAP FOR {self.name}-{identifier}...")
        if self.rpm_map is None:
            if isinstance(job, str) and job == 'full':
                group_focus = None
            else:
                group_focus, _ = job
            self.build_rpm_map(subset_df, group_focus)
        log_print(f"STARTING TESTS FOR {self.name}-{identifier}...\n")
        if id == 0:  # t-test job
            result = self.process_bioproject_other()
        else:
            if isinstance(job, str):
                result = self.process_bioproject_perms(None, self.n_actual_permutation_sets)
            else:
                result = self.process_bioproject_perms(job[0], job[1])

        log_print(f"FINISHED PROCESSING {self.name}-{identifier} in {round(time.time() - time_start, 3)} seconds\n")
        return result

    def get_test_stats(self, index_is_include, index_list, rpm_map, group_name, attributes, values) \
            -> tuple[float, float, float, float, float, float, list, list, Any] | None:
        """get the test stats for the given group"""
        # could be optimized?
        true_rpm, false_rpm = [], []
        for i, rpm_val in enumerate(rpm_map):
            if not IMPLICIT_ZEROS and rpm_val == MAP_UNKNOWN:
                continue
            if (i in index_list and index_is_include) or (i not in index_list and not index_is_include):
                true_rpm.append(rpm_val)
            else:
                false_rpm.append(rpm_val)

        num_true, num_false = len(true_rpm), len(false_rpm)

        if num_true < 2 or num_false < 2:  # this probably only might happen if IMPLIED_ZEROS is False
            log_print(f'skipping {group_name} - {attributes}:{values} '
                      f'because num_true or num_false < 2', 2)
            return None

        # calculate desecriptive stats
        # NON CORRECTED VALUES
        mean_rpm_true = np.nanmean(true_rpm)
        mean_rpm_false = np.nanmean(false_rpm)
        sd_rpm_true = np.nanstd(true_rpm)
        sd_rpm_false = np.nanstd(false_rpm)

        # skip if both conditions have 0 reads (this should never happen, but just in case)
        if mean_rpm_true == mean_rpm_false == 0:
            return None

        # calculate fold change
        def get_log_fold_change(true, false):
            """calculate the log fold change of true with respect to false
                if true and false is 0, then return 0
            """
            if true == 0 and false == 0:
                return 0
            elif true == 0:
                return 'negative inf'
            elif false == 0:
                return 'inf'
            else:
                return np.log2(true / false)

        fold_change = get_log_fold_change(mean_rpm_true, mean_rpm_false)

        return num_true, num_false, mean_rpm_true, mean_rpm_false, sd_rpm_true, sd_rpm_false, true_rpm, false_rpm, fold_change

    def process_bioproject_other(self) -> str | None:
        """Process the given group, and return an output file
            """

        def run_ttests_for_group(shared_results, group):
            """function to be ran in parallel (groups are run in parallel, within a group is in series)"""
            reusable_results = {}
            results = ''
            for _, row in self.metadata_df.iterrows():
                if row['test_type'] == 't_test':
                    new_row = {
                        'include': row['include?'],
                        'biosample_index_list': row['biosample_index_list'],
                        'group_rpm_list': self.rpm_map[group],
                        'group': group,
                        'attributes': row['attributes'],
                        'values': row['values'],
                        'test_id': -1
                    }
                    self.process_set_test(results, new_row, 't_test', reusable_results)
            shared_results[group] = results
            return results

        num_workers = cpu_count()
        with Manager() as manager:
            shared_results = manager.dict()
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(run_ttests_for_group, [(shared_results, group) for group in self.groups])

        log_print(f"Finished processing tests for {self.name}\n")

        # convert the results to a string
        results_str = ''
        for group in self.groups:
            results_str += results[group] + '\n'
        return results_str

    def process_bioproject_perms(self, job_groups: dict | None, num_tests) -> str | None:
        """Process the given bioproject, and return an output file - concatenation of several group outputs
        """
        tests = []
        test_id = 0
        if job_groups is not None:
            groups = job_groups.keys()
        else:
            groups = self.groups
        for group in groups:
            # get subset of metadata_df that are permutation tests and aren't blacklisted fields
            self.metadata_df = self.metadata_df[self.metadata_df['test_type'] == 'permutation-test']
            if not TEST_BLACKLISTED_METADATA_FIELDS:
                self.metadata_df = self.metadata_df[self.metadata_df['skippable?'] == 0]

            if job_groups is not None:
                start, end = job_groups[group]
                sets_subset_df = self.metadata_df.iloc[start:end]
            else:
                sets_subset_df = self.metadata_df

            # add the tests to the tests list
            for _, row in sets_subset_df.iterrows():
                tests.append({
                    'include': row['include?'],
                    'biosample_index_list': row['biosample_index_list'],
                    'group_rpm_list': self.rpm_map[group],
                    'group': group,
                    'attributes': row['attributes'],
                    'values': row['values'],
                    'test_id': test_id
                })
                test_id += 1
        del self.metadata_df

        num_workers = self.num_conc_procs
        with Manager() as manager:
            shared_results = manager.dict()  # Shared dictionary
            reusable_keys = manager.dict()
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(self.process_set_test, [(shared_results, test, 'permutation_test', reusable_keys) for test in tests])

        log_print(f"Finished processing tests for {self.name}\n")

        # convert the results to a string
        results_str = ''
        for result in results:
            results_str += results[result] + '\n'
        return results_str

    def process_set_test(self, results: dict | str, row: dict, test_type: str, reusable_results: dict) -> None:
        """Process a single set of permutation tests
        if test_type is t_test then results is a string,
        if test_type is permutation_test then results is a dictionary
        """
        test_start_time = time.time()

        group, rpm_list = row['group'], row['group_rpm_list']
        index_is_inlcude, index_list = row['include'], row['biosample_index_list']
        test_key = self.get_test_stats(index_is_inlcude, index_list, rpm_list, group, row['attributes'], row['values'])
        if test_key is None:
            return
        num_true, num_false, mean_rpm_true, mean_rpm_false, sd_rpm_true, sd_rpm_false, true_rpm, false_rpm, fold_change = test_key
        test_key = test_key + (group,)
        if test_key in reusable_results and (mean_rpm_false == 0 or mean_rpm_true == 0):
            test_statistic, p_value, status = reusable_results[test_key]
            log_print(f"Reusing results for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}", 2)
        else:
            log_print(f"Running {test_type} for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}", 2)
            try:
                if test_type == 't_test':
                    # T TEST
                    assert min(num_false, num_true) < 4
                    status = 't_test'
                    test_statistic, p_value = ttest_ind_from_stats(mean1=mean_rpm_true, std1=sd_rpm_true, nobs1=num_true, mean2=mean_rpm_false, std2=sd_rpm_false, nobs2=num_false,
                                                                   equal_var=False)
                    reusable_results[test_key] = (fold_change, test_statistic, p_value, status)

                elif test_type == 'permutation_test':
                    # PERMUTATION TEST
                    assert min(num_false, num_true) >= 4
                    status = 'permutation_test'
                    num_samples = 10000  # note, we do not need to lower this to be precise (using n choose k) since scipy does this for us anyway

                    def mean_diff_statistic(x, y, axis):
                        """if -ve, then y is larger than x"""
                        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

                    res = permutation_test((true_rpm, false_rpm), statistic=mean_diff_statistic, n_resamples=num_samples, vectorized=True)
                    p_value, test_statistic = res.pvalue, res.statistic

                else:
                    log_print(f"Error: unknown test type {test_type}", 2)
                    return
                reusable_results[test_key] = (fold_change, test_statistic, p_value, status)
            except Exception as e:
                log_print(f"Error running permutation test for {row['group']} - {row['attributes']}:{row['values']} - {e}", 2)
                return

        if p_value < P_VALUE_THRESHOLD:
            status += '; significant'
            too_many, threshold = 'too many biosamples to list', 200
            true_biosamples = '; '.join([self.metadata_ref_lst[i] for i in index_list]) \
                if (num_true < threshold and index_is_inlcude) or (num_false < threshold and not index_is_inlcude) else too_many
            false_biosamples = '; '.join([self.metadata_ref_lst[i] for i in range(len(self.metadata_ref_lst)) if i not in index_list]) \
                if (num_true < threshold and not index_is_inlcude) or (num_false < threshold and index_is_inlcude) else too_many
            if not index_is_inlcude:
                true_biosamples, false_biosamples = false_biosamples, true_biosamples
        else:
            true_biosamples, false_biosamples = '', ''
        log_print(f"Finished with p-value: {p_value}", 2)

        test_end_time = time.time() - test_start_time
        log_print(f"Test took {test_end_time} seconds", 2)

        # record the output
        this_result = (
            f"{self.name},{row['group']},{row['attributes'].replace(',', ' ')},{row['values'].replace(',', ' ')},{status},"
            f"{num_true},{num_false},{mean_rpm_true},{mean_rpm_false},{sd_rpm_true},{sd_rpm_false},{fold_change},"
            f"{test_statistic},{p_value},{true_biosamples},{false_biosamples},{test_end_time}\n"
        )
        log_print(this_result, 2)
        if row['test_id'] == -1:
            results += this_result
        else:
            results[row['test_id']] = this_result


def get_table_via_sql_query(connection: dict, query: str, accs: list[str]) -> pd.DataFrame | None:
    """Get the data from the given table using the given query with the given accessions
    """
    try:
        conn = psycopg2.connect(**connection)
        conn.close()  # close the connection because we are only checking if we can connect
        log_print(f"Successfully connected to database at {connection['host']}")
    except psycopg2.Error:
        log_print(f"Unable to connect to database at {connection['host']}")

    accs_str = ", ".join([f"'{acc}'" for acc in accs])

    with psycopg2.connect(**connection) as conn:
        try:
            df = pd.read_sql(query % accs_str, conn)
            return df
        except psycopg2.Error:
            log_print("Error in executing query")
            return None


def preprocessing(data_file: pd.DataFrame, start_time: float) -> tuple[int, int, int] | None:
    """Run MWAS on the given data file
    input_info is a tuple of the group and quantifier column names
    storage will usually be the tmpfs mount point mount_tmpfs
    """
    problematic_biopjs_file_actual = f"{TEMP_LOCAL_BUCKET}/{PROBLEMATIC_BIOPJS_FILE}"
    date = time.asctime().replace(' ', '_').replace(':', '-')
    log_print(f"Starting MWAS at {date}")

    # ================
    # PRE-PROCESSING
    # ================
    prep_start_time = time.time()

    update_progress('Preprocessing', 0, 0, 0, 0, start_time)

    # CREATE BIOSAMPLES DATAFRAME
    runs = data_file['run'].unique()
    main_df = get_table_via_sql_query(LOGAN_CONNECTION_INFO,
                                      LOGAN_SRA_QUERY[0] if not ALREADY_NORMALIZED else LOGAN_SRA_QUERY[1], runs)
    if main_df is None:
        return
    if not ALREADY_NORMALIZED:
        main_df['spots'] = main_df['spots'].replace(0, NORMALIZING_CONST)

    # merge the main_df with the input_df (must assume both have run column)
    main_df = main_df.merge(data_file, on='run', how='outer')
    main_df.fillna(MAP_UNKNOWN, inplace=True)
    del data_file, runs
    main_df.groupby('bio_project')

    # BIOPROJECTS DATA TABLE & JOB STRUCTURE SCHEDULING & TIME + SPACE ESTIMATIONS & CREATE BIOPROJECT LAMBDA JOBS
    bioprojects_list = main_df['bio_project'].unique()

    biopj_info_df = get_table_via_sql_query(SERRATUS_CONNECTION_INFO, META_QUERY, bioprojects_list)
    # get subset of df where comment_code does not have 1 in first bit. Note comment_code is integer but it's actually binary mask
    ignore_subset = biopj_info_df[~biopj_info_df['comment_code'].isin([1, 32])]

    # record the bioprojects that were ignored onto the problematic bioprojects file
    with open(problematic_biopjs_file_actual, 'a') as f:
        # add bioprojects that are in main_df but are not in biopj_info_df to the problematic bioprojects file
        for bioproject in bioprojects_list:
            if bioproject not in biopj_info_df['bioproject'].values:
                f.write(f"{bioproject},no_metadata_currently_available_in_mwas_database\n")
        for _, row in ignore_subset.iterrows():
            reason = decode_comment_code(row['comment_code'])
            f.write(f"{row['bioproject']},{reason}\n")

    del ignore_subset
    biopj_info_df = biopj_info_df[biopj_info_df['comment_code'].isin([1, 32])]
    # remove rows where the bioproject's test size will make it impossible to run on any lambda due to memory constraints, which is 7 GB
    rejected = biopj_info_df[biopj_info_df['n_biosamples'] * 240000 > MAX_RAM_SIZE]
    biopj_info_df = biopj_info_df[biopj_info_df['n_biosamples'] * 240000 <= MAX_RAM_SIZE]
    with open(problematic_biopjs_file_actual, 'a') as f:
        for _, row in rejected.iterrows():
            f.write(f"{row['bioproject']},blacklisted - too_large_to_run_on_lambda\n")
    del rejected

    # for each bioproject in main_df, count the number of unique groups, and then number of groups that have less than X rows in main_df
    # implying they should be skipped. Also add the bioproject to rejected if all groups are skipped
    def calculate_groups_and_skipped_groups(df):
        """Calculate the number of groups and skipped groups for a bioproject"""
        num_groups = df['group'].nunique()
        num_skipped_groups = (df['group'].value_counts() < GROUP_NONZEROS_ACCEPTANCE_THRESHOLD).sum()
        return pd.Series([num_groups, num_skipped_groups], index=['n_groups', 'n_skipped_groups'])

    # Calculate number of groups and skipped groups for each bioproject
    group_info = main_df.groupby('bio_project').apply(calculate_groups_and_skipped_groups).reset_index()
    biopj_info_df = biopj_info_df.merge(group_info, left_on='bioproject', right_on='bio_project', how='left')
    # remove redundant column
    biopj_info_df.drop('bio_project', axis=1, inplace=True)
    # Identify bioprojects where all groups are skipped
    all_groups_skipped = biopj_info_df[biopj_info_df['n_groups'] == biopj_info_df['n_skipped_groups']]['bioproject']
    # Write all_groups_skipped to file
    with open(problematic_biopjs_file_actual, 'a') as f:
        for bioproject in all_groups_skipped:
            f.write(f"{bioproject},all_groups_skipped\n")
    # Remove all_groups_skipped bioprojects from biopj_info_df
    biopj_info_df = biopj_info_df[biopj_info_df['n_groups'] != biopj_info_df['n_skipped_groups']]

    num_bioprojects = len(biopj_info_df)

    # create bioproject objects for each bioproject in biopj_info_df and dispatch the lambdas and store them in a dictionary
    bioprojects_dict = {}
    total_perm_tests = total_lambda_jobs = 0
    for bioproject_name, row_srs in biopj_info_df.groupby('bioproject'):
        bioproject_obj = BioProjectInfo(bioproject_name, row_srs['condensed_md_file_size'].iloc[0], row_srs['n_biosamples'].iloc[0], row_srs['n_sets'].iloc[0],
                                        row_srs['n_permutation_sets'].iloc[0], 0,  # row_srs['n_skippable_permutation_sets'].iloc[0],
                                        row_srs['n_groups'].iloc[0], row_srs['n_skipped_groups'].iloc[0])

        n_perm_tests, lambda_jobs = bioproject_obj.batch_lambda_jobs(main_df, 60)
        total_perm_tests += n_perm_tests
        total_lambda_jobs += lambda_jobs + 1  # +1 for the non-perm test lambda
        bioprojects_dict[bioproject_name] = bioproject_obj
        log_print(f"{bioproject_name} had {lambda_jobs} lambda jobs and {n_perm_tests} permutation tests")

    log_print(f"Number of bioprojects to process: {num_bioprojects}, Number of ignored bioprojects: {len(bioprojects_list) - num_bioprojects}, "
              f"Number of lambda jobs: {total_lambda_jobs}, Number of permutation tests: {total_perm_tests}")

    log_print(f"preprocessing time: {time.time() - prep_start_time}", 1)

    # pickle the main_df and store it in the bioproject's s3 dir
    main_df.to_pickle(f'{TEMP_LOCAL_BUCKET}/temp_main_df.pickle')
    main_df_s3_link = f"{S3_OUTPUT_DIR}/temp_main_df.pickle"

    update_progress('Dispatching Lambdas', num_bioprojects, total_lambda_jobs, total_perm_tests, 0, start_time)
    del biopj_info_df, main_df

    # ================
    # PROCESSING
    # ================

    # TODO: invoke SNS and tell it to start listening for the results and how many results to expect
    LAMBDA_CLIENT = boto3.client('lambda')
    for bioproject in bioprojects_dict:
        bioproject_obj = bioprojects_dict[bioproject]
        bioproject_obj.dispatch_all_lambda_jobs(main_df_s3_link, LAMBDA_CLIENT)
        del bioproject_obj

    return num_bioprojects, total_lambda_jobs, total_perm_tests


def main(args: list[str]) -> int | None | tuple[int, str]:
    """Main function to run MWAS on the given data file"""
    num_args = len(args)
    time_start = time.time()

    # Check if the correct number of arguments is provided
    if num_args < 2 or args[1] in ('-h', '--help'):
        return 1, "Usage: python mwas_general.py data_file.csv [flags]"
    elif args[1].endswith('.csv'):

        # s3 storing
        try:
            global TEMP_LOCAL_BUCKET, logger

            hash_dest = args[args.index('--s3') + 1]
            # check if the hash_dest exists in our s3 bucket already (if it does, exit mwas)
            process = subprocess.run(SHELL_PREFIX + f"s5cmd ls {S3_OUTPUT_DIR}{hash_dest}/", shell=True, stderr=subprocess.PIPE)
            if not process.stderr:
                # this implies we found something successfully via ls, so we should exit
                return 0, 'This input has already been processed. Please refer to the s3 bucket for the output.'

            # create local disk folder to sync with s3
            TEMP_LOCAL_BUCKET = f"./{hash_dest}"
            problematic_biopjs_file_actual = f"{TEMP_LOCAL_BUCKET}/{PROBLEMATIC_BIOPJS_FILE}"
            preproc_log_file_actual = f"{TEMP_LOCAL_BUCKET}/{PREPROC_LOG_FILE}"
            output_file_actual = f"{TEMP_LOCAL_BUCKET}/{OUTPUT_CSV_FILE}"
            if os.path.exists(TEMP_LOCAL_BUCKET):
                rmtree(TEMP_LOCAL_BUCKET)
            os.mkdir(TEMP_LOCAL_BUCKET)
            # make subfolders for outputs and logs
            os.mkdir(f"{TEMP_LOCAL_BUCKET}/{OUTPUT_FILES_DIR}")
            os.mkdir(f"{TEMP_LOCAL_BUCKET}/{PROC_LOG_FILES_DIR}")

            # create the problematic bioprojects file and logger and progress.json
            logger = logging.getLogger(preproc_log_file_actual)
            if logger.hasHandlers():
                logger.handlers.clear()
            fh = logging.FileHandler(preproc_log_file_actual)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
            with open(problematic_biopjs_file_actual, 'w') as f:
                f.write('bioproject,reason\n')
            update_progress('Starting', 0, 0, 0, 0, time_start, False)

            # create the s3 bucket
            process = subprocess.run(SHELL_PREFIX + f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}", shell=True)
            if process.returncode == 0:
                log_print(f"Created s3 bucket: {S3_OUTPUT_DIR}")
            else:
                log_print(f"Error in creating s3 bucket: {S3_OUTPUT_DIR}", 0)
                return 1, 'could not create the s3 bucket'
        except Exception as e:
            log_print(f"Error in setting s3 output directory: {e}", 0)
            return 1, 'could not set s3 output directory'

        # set flags
        global logging_level, IMPLICIT_ZEROS, GROUP_NONZEROS_ACCEPTANCE_THRESHOLD, ALREADY_NORMALIZED, P_VALUE_THRESHOLD
        if '--suppress-logging' in args:
            logging_level = 1
        if '--no-logging' in args:
            logging_level = 0
        if '--explicit-zeros' in args or '--explicit-zeroes' in args:
            IMPLICIT_ZEROS = False
        if '--already-normalized' in args:  # TODO: test
            ALREADY_NORMALIZED = True
        if '--p-value-threshold' in args:  # TODO: test
            try:
                P_VALUE_THRESHOLD = float(args[args.index('--p-value-threshold') + 1])
            except Exception as e:
                log_print(f"Error in setting p-value threshold: {e}", 0)
                return 1
        if '--group-nonzero-threshold' in args:  # TODO: test
            try:
                GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = int(args[args.index('--group-nonzero-threshold') + 1])
            except Exception as e:
                log_print(f"Error in setting group nonzeros threshold: {e}", 0)
                return 1

        try:  # reading the input file
            input_df = pd.read_csv(args[1])  # arg1 is a file path
            # rename group and quantifier columns to standard names, and also save original names
            group_by, quantifying_by = input_df.columns[1], input_df.columns[2]
            input_df.rename(columns={
                input_df.columns[0]: 'run', input_df.columns[1]: 'group', input_df.columns[2]: 'quantifier'
            }, inplace=True)

            # assume it has three columns: run, group, quantifier. And the group and quantifier columns have special names
            if len(input_df.columns) != 3:
                log_print("Data file must have three columns in this order: <run>, <group>, <quantifier>", 0)
                return 1, 'invalid data file'

            # attempt to correct column types so the next if block doesn't exit us
            input_df['run'] = input_df['run'].astype(str)
            input_df['group'] = input_df['group'].astype(str)
            input_df['quantifier'] = pd.to_numeric(input_df['quantifier'], errors='coerce')

            # check if run and group contain string values and quantifier contains numeric values
            if (input_df['run'].dtype != 'object' or input_df['group'].dtype != 'object'
                    or input_df['quantifier'].dtype not in ('float64', 'int64')):
                log_print("run and group column must contain string values, and quantifier column must contain numeric values", 0)
                return 1
        except FileNotFoundError:
            log_print("File not found", 0)
            return 1, 'file not found'

        # RUN MWAS
        log_print("RUNNING MWAS...", 0)
        num_bioprojects, num_lambda_jobs, num_permutation_tests = preprocessing(input_df, time_start)
        log_print(f"Time taken: {time.time() - time_start} minutes", 0)

        # create header for output file
        with open(output_file_actual, 'w') as f:
            header = ''
            for word in OUT_COLS:
                if word == 'group':
                    header += f"{group_by},"
                else:
                    header += word + ','
            f.write(header[:-1] + '\n')
        update_progress('Processing', num_bioprojects, num_lambda_jobs, num_permutation_tests, 0, time_start)

        # remove the local copy of the s3 bucket
        rmtree(TEMP_LOCAL_BUCKET)
        print(f"Removed local copy of s3 bucket: {TEMP_LOCAL_BUCKET}")

        return 0, f'Preprocessing completed in {time.time() - time_start} seconds'
    else:
        log_print("Invalid arguments", 0)
        return 1, 'invalid arguments'


def update_progress(status, num_bioprojects, num_lambda_jobs, num_permutation_tests, num_jobs_completed, start_time, sync_s3=True) -> None:
    """Update the progress file with the given progress dictionary"""
    with open(f"{TEMP_LOCAL_BUCKET}/{PROGRESS_FILE}", 'w') as f:
        json.dump({
            'status': status,
            'num_bioprojects': str(num_bioprojects),
            'num_lambda_jobs': str(num_lambda_jobs),
            'num_permutation_tests': str(num_permutation_tests),
            'num_jobs_completed': str(num_jobs_completed),
            'time_elapsed': str(time.time() - start_time)
        }, f)
    if sync_s3:
        s3_sync()


def s3_sync():
    """Sync the local s3 bucket with the s3 bucket"""
    process = subprocess.run(SHELL_PREFIX + f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}", shell=True)
    if process.returncode == 0:
        log_print(f"Synced s3 bucket: {S3_OUTPUT_DIR}")
    else:
        log_print(f"Error in syncing s3 bucket: {S3_OUTPUT_DIR}", 0)


if __name__ == '__main__':
    exit_code = main(sys.argv)
    sys.exit(exit_code)
