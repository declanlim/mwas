"""Generalized MWAS
"""
import math
# built-in libraries
import os
import sys
import platform
import subprocess
import pickle
import time
import tracemalloc
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



# system & path constants
PICKLE_DIR = './pickles'  # will be relative to working directory
OUTPUT_DIR_DISK = './outputs'
S3_METADATA_DIR = 's3://serratus-biosamples/condensed-bioproject-metadata'  # 's3://serratus-biosamples/mwas_setup/bioprojects'
S3_OUTPUT_DIR = None
TEMP_LOCAL_BUCKET = None
PROBLEMATIC_BIOPJS_FILE = 'bioprojects_ignored.txt'
OS = platform.system()
SHELL_PREFIX = 'wsl ' if OS == 'Windows' else ''
DEFAULT_MOUNT_POINT = '/mnt/mwas'
TMPFS_DIR = '/mnt/tmpfs'
SYSTEM_MOUNTS = 'wmic logicaldisk get name' if OS == 'Windows' else 'mount'

# processing constants
BLOCK_SIZE = 1000
MAX_RAM_SIZE = 7 * 1024 ** 3  # 7 GB
MAX_PROJECT_SIZE = 50 * 1024 ** 2  # 50 MB
PICKLE_SPACE_LIMIT = 2 * 1024 ** 3  # 2 GB
PARALLELIZING = False

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

# flags
IMPLICIT_ZEROS = True  # TODO: implement this flag for when it's False
GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = 3  # if group has at least this many nonzeros, then it's okay. Note, this flag is only used if IMPLICIT_ZEROS is True
ALREADY_NORMALIZED = False  # if it's false, then we need to normalize the data by dividing quantifier by spots
P_VALUE_THRESHOLD = 0.005
ONLY_T_TEST = False  # if True, then only t tests will be run, and other tests, e.g. permutation tests, will not be done
COMBINE_OUTPUTS = True  # if false, there will ba a separate output file for each bioproject
PERFOMANCE_STATS = False
INCLUDE_SKIPPED_STATS = False  # if True, then the output will include the stats of the skipped tests
PRE_CALC_RPM_MAP = False  # if True, then the rpm map will be precalculated for each bioproject

# constants
MAP_UNKNOWN = 0  # maybe np.nan
NORMALIZING_CONST = 1000000  # 1 million

# OUT_COLS = ['bioproject', 'group', 'metadata_field', 'metadata_value', 'status', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
#             'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value']
OUT_COLS_STR = """bioproject,%s,metadata_field,metadata_value,status,%snum_true,num_false,mean_rpm_true,mean_rpm_false,sd_rpm_true,sd_rpm_false,fold_change,test_statistic,p_value,true_biosamples,false_biosamples"""

# debugging
# num_tests = 0
# progress = 0
logging_level = 2  # 0: no logging, 1: minimal logging, 2: verbose logging
use_logger = False
logging.basicConfig(level=logging.INFO)
# refresh the log file
with open("mwas_logging.txt", 'w') as f:
    pass
logger = logging.getLogger("mwas_logging.txt")
if logger.hasHandlers():
    logger.handlers.clear()
fh = logging.FileHandler("mwas_logging.txt")
formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


def log_print(msg: Any, lvl: int = 1) -> None:
    """Print a message if the logging level is appropriate"""
    global logging_level, use_logger
    if lvl <= logging_level:
        if use_logger:
            logger.info(msg)
        else:
            print(msg)


def sync_s3() -> None:
    """Sync the local output directory with the given s3 bucket"""
    if S3_OUTPUT_DIR:
        # update progress report
        log_print("Syncing the local output directory with the given s3 bucket...")
        # check if progress_report.json exists
        with open(f'{TEMP_LOCAL_BUCKET}/progress_report.json', 'w') as f:
            PROGRESS.update_time()
            json_data = PROGRESS.get_progress()
            json.dump(json_data, f)

        # sync the local output directory with the given s3 bucket
        try:
            subprocess.run(SHELL_PREFIX + f"cp mwas_logging.txt {TEMP_LOCAL_BUCKET}", shell=True)
        except Exception as e:
            log_print(f"Error in copying the log file to the local bucket: {e}")
        try:
            command = f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}"
            subprocess.run(SHELL_PREFIX + command, shell=True)
        except Exception as e:
            log_print(f"Error in storing the combined output file in s3: {e}")


class Progress:
    """MWAS progress stats"""

    def __init__(self) -> None:
        self.progress = 0.0
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.remaining_time = 0.0
        self.init_est_time = -1
        self.init_est_num_tests = -1
        self.num_tests_done = 0
        self.num_sig_results = 0
        self.mwas_processing_stage = 'Pre-processing'
        self.num_bioprojects = 0
        self.num_bioprojects_done = 0
        self.num_ignored_bioprojects = 0
        self.estimated_already = False

    def set_estimates(self, init_est_time: float, init_est_num_tests: int) -> None:
        """Set the initial estimates for the progress stats"""
        self.init_est_time = init_est_time
        self.init_est_num_tests = init_est_num_tests
        self.estimated_already = True

    def update_time(self) -> None:
        """Update the small progress stats"""

        def seconds_to_minutes(seconds: float) -> float:
            return round(seconds / 60, 2)

        self.elapsed_time = seconds_to_minutes(time.time() - self.start_time)
        self.remaining_time = seconds_to_minutes(self.init_est_time - self.elapsed_time) if self.init_est_time > 0 else 0

    def update_small_progress(self, num_tests_done: int, num_sig_results: int) -> None:
        """Update the small progress stats"""
        self.num_tests_done = num_tests_done
        self.num_sig_results = num_sig_results
        self.progress = round(100 * num_tests_done / self.init_est_num_tests, 2)

    def update_large_progress(self, mwas_processing_stage: str,
                              num_bioprojects: int, num_bioprojects_done: int, num_ignored_bioprojects: int) -> None:
        """Update the large progress stats"""
        self.mwas_processing_stage = mwas_processing_stage
        if self.mwas_processing_stage == 'Completed':
            self.remaining_time = 0
            self.progress = 100
        self.num_bioprojects = num_bioprojects
        self.num_bioprojects_done = num_bioprojects_done
        self.num_ignored_bioprojects = num_ignored_bioprojects

    def get_progress(self) -> dict[str, Any]:
        """Get the progress stats. make sure these match the keys in mwas.sh"""
        return {
            'percent_complete': f"{self.progress}%",
            'elapsed_time': f"{self.elapsed_time} minutes",
            'remaining_time': f"{self.remaining_time} minutes" if self.remaining_time > 0 else 'calculating...',
            'initial_time_estimate': f"{self.init_est_time} minutes" if self.init_est_time > 0 else 'calculating...',
            'initial_num_tests': self.init_est_num_tests if self.init_est_num_tests > 0 else 'calculating...',
            'num_tests_completed': self.num_tests_done,
            'num_sig_results': self.num_sig_results,
            'mwas_processing_stage': self.mwas_processing_stage,
            'bioprojects_processed': f"{self.num_bioprojects_done}/{self.num_bioprojects}",
            'bioprojects_ignored': self.num_ignored_bioprojects
        }


PROGRESS = Progress()


class BioProjectInfo:
    """Class to store information about a bioproject"""

    def __init__(self, name: str, metadata_file_size: int, n_biosamples: int,
                 n_sets: int, n_permutation_sets: int, n_skippable_permutation_sets: int, n_groups: int, n_skipped_groups: int) -> None:
        self.name = name
        self.md_file_size = metadata_file_size
        self.n_sets = n_sets
        self.n_permutation_sets = n_permutation_sets
        self.n_skippable_permutation_sets = n_skippable_permutation_sets
        self.n_actual_permutation_sets = n_permutation_sets - n_skippable_permutation_sets if not INCLUDE_SKIPPED_STATS else n_permutation_sets
        self.n_biosamples = n_biosamples
        self.n_groups = n_groups
        self.n_skipped_groups = n_skipped_groups
        self.metadata_df = None
        self.metadata_ref_lst = None
        self.metadata_path = None
        self.rpm_map = None
        self.num_lambda_jobs = 0
        self.num_conc_procs = 0
        self.groups = []
        self.jobs = []

    def batch_lambda_jobs(self, input_df: pd.DataFrame, time_constraint: int) -> tuple[int, int]:
        """
        time_constraint: in seconds
        """
        LAMBDA_RANGE = (128, 10240)
        LAMBDA_CPU_CORES = 6
        PROCESS_OVERHEAD_BOUND = 84000000  # 84 MB (run module_size_test to determine this, then round up a bit)

        # choosing lambda size
        mem_width = self.n_biosamples * 240000 + PROCESS_OVERHEAD_BOUND
        n_conc_procs = LAMBDA_CPU_CORES
        while mem_width * n_conc_procs > LAMBDA_RANGE[1]:
            n_conc_procs -= 1
        if n_conc_procs == 0:
            print(f"Error: not enough memory to run a single test on a lambda functions for bio_project {self.name}")
            return 0, 0
        self.num_conc_procs = n_conc_procs

        # est_series_time = time_per_test * self.n_permutation_sets * num_actual_groups
        # print(f"Estimated worst-case time if ran tests in series: {est_series_time} seconds")

        time_per_test = self.n_biosamples * 0.0003
        total_tests = (self.n_groups - self.n_skipped_groups) * self.n_actual_permutation_sets
        lambda_area = (math.floor(time_constraint / time_per_test) * n_conc_procs)
        self.num_lambda_jobs = math.ceil(total_tests / lambda_area)

        self.jobs = []
        self.groups = [g for g in input_df['group'].unique()
                       if input_df['group'].value_counts()[g] > GROUP_NONZEROS_ACCEPTANCE_THRESHOLD]

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

        print(self.jobs)  # jobs are of the form: {'group1': (start, end), 'group2': (start, end), ...}
        assert self.num_lambda_jobs == len(self.jobs)
        return self.n_actual_permutation_sets, self.num_lambda_jobs

    def dispatch_all_lambda_jobs(self, input_df: pd.DataFrame, lam_client: boto3.client) -> None:
        """lam_client: lambda_client = boto3.client('lambda')

        note lambda handler will call self.process_bioproject(...)
        """
        bioproject_info = self.__dict__
        input_dict = input_df.to_dict(orient='records')

        print(f"dispatching all {self.num_lambda_jobs} lambda jobs for {self.name}")
        # non permutation test lambda(TODO: lambda[s]?)
        lam_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps({
                'bioproject_info': bioproject_info,
                'input_df': input_dict,
                'job_window': {},  # empty implies all groups
                'id': 0  # 0 implies this is the non-perm tests lambda
            })
        )

        for i, job in enumerate(self.jobs):
            # Invoke the Lambda function
            lam_client.invoke(
                FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps({
                    'bioproject_info': bioproject_info,
                    'input_df': input_dict,
                    'job_window': job,
                    'id': i + 1
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
                    print("Warning: metadata ref list and metadata df are not the same length. Requires updating mwas metadata table")
                self.n_biosamples = len(self.metadata_ref_lst)
            os.remove('temp_file.pickle')
        except Exception as e:
            log_print(f"Error in loading metadata for {self.name}: {e}", 2)

    def build_rpm_map(self, input_df: pd.DataFrame, groups_focus: dict) -> None:
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
            if job is None:
                groups = None
            else:
                groups, _ = job
            self.build_rpm_map(subset_df, groups)
        log_print(f"STARTING TESTS FOR {self.name}...\n")
        if job is None:
            result = self.process_bioproject_other()
        else:
            result = self.process_bioproject_perms(job[0], job[1])

        log_print(f"FINISHED processing {self.name} in {round(time.time() - time_start, 3)} seconds\n")
        return result

    def process_bioproject_other(self) -> str | None:
        """Process the given group, and return an output file
            """
        result = ''
        skip_tests = False
        reusable_results = {}  # save results while a group is processed, so we can avoid recomputing them
        for group in self.groups:
            if self.rpm_map[group][1]:  # if the group should be skipped
                if INCLUDE_SKIPPED_STATS:
                    skip_tests = True
                else:
                    continue
            for _, row in self.metadata_df.iterrows():
                test_start_time = time.time()

                index_is_inlcude = row['include?']
                index_list = row['biosample_index_list']

                # could be optimized?
                true_rpm, false_rpm = [], []
                for i, rpm_val in enumerate(self.rpm_map):
                    if not IMPLICIT_ZEROS and rpm_val == MAP_UNKNOWN:
                        continue
                    if (i in index_list and index_is_inlcude) or (i not in index_list and not index_is_inlcude):
                        true_rpm.append(rpm_val)
                    else:
                        false_rpm.append(rpm_val)

                num_true, num_false = len(true_rpm), len(false_rpm)

                if num_true < 2 or num_false < 2:  # this probably only might happen if IMPLIED_ZEROS is False
                    log_print(f'skipping {group} - {row["attributes"]}:{row["values"]} '
                              f'because num_true or num_false < 2', 2)
                    continue

                # calculate desecriptive stats
                # NON CORRECTED VALUES
                mean_rpm_true = np.nanmean(true_rpm)
                mean_rpm_false = np.nanmean(false_rpm)
                sd_rpm_true = np.nanstd(true_rpm)
                sd_rpm_false = np.nanstd(false_rpm)

                # skip if both conditions have 0 reads (this should never happen, but just in case)
                if mean_rpm_true == mean_rpm_false == 0:
                    continue

                # get list of all attributes sep by ; delim
                fields = row['attributes'].split('; ')
                if skip_tests or all(field in BLACKLISTED_METADATA_FIELDS for field in fields):
                    if INCLUDE_SKIPPED_STATS:
                        continue
                    fold_change, test_statistic, p_value = '', '', ''
                    true_biosamples, false_biosamples = '', ''
                    status = 'skipped_statistical_testing'
                else:
                    test_key = (num_true, num_false, mean_rpm_true, mean_rpm_false, sd_rpm_true, sd_rpm_false)
                    if test_key in reusable_results and (mean_rpm_false == 0 or mean_rpm_true == 0):
                        fold_change, test_statistic, p_value, status = reusable_results[test_key]
                        log_print(
                            f"Reusing results for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}",2)
                    else:
                        # calculate fold change and check if any values are nan
                        fold_change = get_log_fold_change(mean_rpm_true, mean_rpm_false)

                        # if there are at least 4 values in each group, run a permutation test, otherwise run a t test
                        try:
                            log_print(
                                f"Running statistical test for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}", 2)
                            if min(num_false, num_true) < 4 or ONLY_T_TEST:
                                # scipy t test
                                status = 't_test'
                                test_statistic, p_value = ttest_ind_from_stats(mean1=mean_rpm_true, std1=sd_rpm_true, nobs1=num_true,
                                                                               mean2=mean_rpm_false, std2=sd_rpm_false, nobs2=num_false,
                                                                               equal_var=False)
                            else:
                                # run a permutation test
                                status = 'permutation_test'
                                num_samples = 10000  # note, we do not need to lower this to be precise (using n choose k) since scipy does this for us anyway
                                res = permutation_test((true_rpm, false_rpm), statistic=mean_diff_statistic, n_resamples=num_samples,
                                                       vectorized=True)
                                p_value, test_statistic = res.pvalue, res.statistic
                        except Exception as e:
                            log_print(f'Error running statistical test for {group} - {row["attributes"]}:{row["values"]} - {e}', 2)
                            continue

                        reusable_results[test_key] = (fold_change, test_statistic, p_value, status)

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

                if PERFOMANCE_STATS:
                    _, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    extra_info = f'{time.time() - test_start_time}, {peak_memory},'
                else:
                    extra_info = ''

                # record the output
                this_result = (
                    f"{self.name},{group},{row['attributes'].replace(',', ' ')},{row['values'].replace(',', ' ')},{status},{extra_info}"
                    f"{num_true},{num_false},{mean_rpm_true},{mean_rpm_false},{sd_rpm_true},{sd_rpm_false},{fold_change},{test_statistic},{p_value},{true_biosamples},{false_biosamples}\n")

                log_print(this_result, 2)
                shared_dict[row['test_id']] = this_result
            return result

    def process_set_ttest(self, shared_dict: dict, row: dict) -> None:
        """Process a single set of t tests
        """
        test_start_time = time.time()

        index_is_inlcude, index_list = row['include'], row['biosample_index_list']
        group_name, group_rpm_lst = row['group'], row['group_rpm_list']

        # could be optimized?
        true_rpm, false_rpm = [], []
        for i, rpm_val in enumerate(group_rpm_lst):
            if not IMPLICIT_ZEROS and rpm_val == MAP_UNKNOWN:
                continue
            if (i in index_list and index_is_inlcude) or (i not in index_list and not index_is_inlcude):
                true_rpm.append(rpm_val)
            else:
                false_rpm.append(rpm_val)

        num_true, num_false = len(true_rpm), len(false_rpm)

        if num_true < 2 or num_false < 2:
            log_print(f'skipping {group_name} - {row["attributes"]}:{row["values"]} '
                      f'because num_true or num_false < 2', 2)
            return

    def process_bioproject_perms(self, job_groups, num_tests) -> str | None:
        """Process the given bioproject, and return an output file - concatenation of several group outputs
        """
        tests = []
        test_id = 0
        for group in job_groups:
            start, end = job_groups[group]
            # get subset of metadata_df that are permutation tests and aren't blacklisted fields
            self.metadata_df = self.metadata_df[self.metadata_df['test_type'] == 'permutation-test']
            self.metadata_df = self.metadata_df[self.metadata_df['skipppable?'] == 0]
            sets_subset_df = self.metadata_df.iloc[start:end]

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

        num_workers = cpu_count()
        with Manager() as manager:
            shared_results = manager.dict()  # Shared dictionary
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(self.process_set_perm, [(shared_results, test) for test in tests])

        log_print(f"Finished processing tests for {self.name}\n")
        return results

    def process_set_perm(self, shared_dict: dict, row: dict) -> None:
        """Process a single set of permutation tests
        """
        test_start_time = time.time()

        index_is_inlcude, index_list = row['include'], row['biosample_index_list']
        group_name, group_rpm_lst = row['group'], row['group_rpm_list']

        # could be optimized?
        true_rpm, false_rpm = [], []
        for i, rpm_val in enumerate(group_rpm_lst):
            if not IMPLICIT_ZEROS and rpm_val == MAP_UNKNOWN:
                continue
            if (i in index_list and index_is_inlcude) or (i not in index_list and not index_is_inlcude):
                true_rpm.append(rpm_val)
            else:
                false_rpm.append(rpm_val)

        num_true, num_false = len(true_rpm), len(false_rpm)

        if num_true < 2 or num_false < 2:  # this probably only might happen if IMPLIED_ZEROS is False
            log_print(f'skipping {group_name} - {row["attributes"]}:{row["values"]} '
                      f'because num_true or num_false < 2', 2)
            return

        # calculate desecriptive stats
        # NON CORRECTED VALUES
        mean_rpm_true = np.nanmean(true_rpm)
        mean_rpm_false = np.nanmean(false_rpm)
        sd_rpm_true = np.nanstd(true_rpm)
        sd_rpm_false = np.nanstd(false_rpm)

        # skip if both conditions have 0 reads (this should never happen, but just in case)
        if mean_rpm_true == mean_rpm_false == 0:
            return

        # calculate fold change and check if any values are nan
        fold_change = get_log_fold_change(mean_rpm_true, mean_rpm_false)

        # if there are at least 4 values in each group, run a permutation test, otherwise run a t test
        try:
            log_print(
                f"Running permutation test for bioproject: {self.name}, group: {group_name}, set: {row['attributes']}:{row['values']}",
                2)
            assert min(num_false, num_true) >= 4
            status = 'permutation_test'
            num_samples = 10000  # note, we do not need to lower this to be precise (using n choose k) since scipy does this for us anyway
            res = permutation_test((true_rpm, false_rpm), statistic=mean_diff_statistic, n_resamples=num_samples,
                                   vectorized=True)
            p_value, test_statistic = res.pvalue, res.statistic
        except Exception as e:
            log_print(f'Error running permutation test for {group_name} - {row["attributes"]}:{row["values"]} - {e}', 2)
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

        if PERFOMANCE_STATS:
            extra_info = f'{time.time() - test_start_time},'
        else:
            extra_info = ''

        # record the output
        this_result = (
            f"{self.name},{group_name},{row['attributes'].replace(',', ' ')},{row['values'].replace(',', ' ')},{status},{extra_info}"
            f"{num_true},{num_false},{mean_rpm_true},{mean_rpm_false},{sd_rpm_true},{sd_rpm_false},{fold_change},{test_statistic},{p_value},{true_biosamples},{false_biosamples}\n"
        )
        log_print(this_result, 2)
        shared_dict[row['test_id']] = this_result


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


def mean_diff_statistic(x, y, axis):
    """if -ve, then y is larger than x"""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def run_on_file(data_file: pd.DataFrame, input_info: tuple[str, str]) -> None:
    """Run MWAS on the given data file
    input_info is a tuple of the group and quantifier column names
    storage will usually be the tmpfs mount point mount_tmpfs
    """
    date = time.asctime().replace(' ', '_').replace(':', '-')
    log_print(f"Starting MWAS at {date}")
    # ================
    # PRE-PROCESSING
    # ================
    prep_start_time = time.time()

    # clear out the blacklist file
    with open(PROBLEMATIC_BIOPJS_FILE, 'w') as f:
        f.write('bioproject,reason\n')

    # CREATE BIOSAMPLES DATAFRAME
    runs = data_file['run'].unique()
    main_df = get_table_via_sql_query(LOGAN_CONNECTION_INFO,
                                      LOGAN_SRA_QUERY[0] if not ALREADY_NORMALIZED else LOGAN_SRA_QUERY[1],
                                      runs)
    if main_df is None:
        return
    if not ALREADY_NORMALIZED:
        main_df['spots'] = main_df['spots'].replace(0, NORMALIZING_CONST)

    # merge the main_df with the input_df (must assume both have run column)
    main_df = main_df.merge(data_file, on='run', how='outer')
    main_df.fillna(MAP_UNKNOWN, inplace=True)
    del data_file, runs
    main_df.groupby('bio_project')

    # TODO: STORE THE MAIN DATAFRAME IN S3
    # temporarily rename the columns to standard names before storing to s3
    # store the main_df in the dir <hash_code> in s3
    # restore the column names back to being general names

    # BIOPROJECTS DATA TABLE & JOB STRUCTURE SCHEDULING & TIME + SPACE ESTIMATIONS & CREATE BIOPROJECT LAMBDA JOBS
    bioprojects_list = main_df['bio_project'].unique()

    biopj_info_df = get_table_via_sql_query(SERRATUS_CONNECTION_INFO, META_QUERY, bioprojects_list)
    # get subset of df where comment_code does not have 1 in first bit. Note comment_code is integer but it's actually binary mask
    ignore_subset = biopj_info_df[~biopj_info_df['comment_code'].isin([1, 32])]

    # record the bioprojects that were ignored onto the problematic bioprojects file
    with open(PROBLEMATIC_BIOPJS_FILE, 'a') as f:
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
    with open(PROBLEMATIC_BIOPJS_FILE, 'a') as f:
        for _, row in rejected.iterrows():
            f.write(f"{row['bioproject']},blacklisted - too_large_to_run_on_lambda\n")
    del rejected

    # for each bioproject in main_df, count the number of unique groups, and then number of groups that have less than X rows in main_df
    # implying they should be skipped. Also add the bioproject to rejected if all groups are skipped
    def calculate_groups_and_skipped_groups(df):
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
    with open(PROBLEMATIC_BIOPJS_FILE, 'a') as f:
        for bioproject in all_groups_skipped:
            f.write(f"{bioproject},all_groups_skipped\n")

    # Remove all_groups_skipped bioprojects from biopj_info_df
    biopj_info_df = biopj_info_df[biopj_info_df['n_groups'] != biopj_info_df['n_skipped_groups']]

    num_bioprojects = len(biopj_info_df)

    # create bioproject objects for each bioproject in biopj_info_df and dispatch the lambdas and store them in a dictionary
    bioprojects_dict = {}
    total_perm_tests = total_lambda_jobs = 0
    LAMBDA_CLIENT = boto3.client('lambda')
    for _, row in biopj_info_df.iterrows():
        bioproject_name = row['bioproject']
        bioproject_obj = BioProjectInfo(bioproject_name, row['condensed_md_file_size'], row['n_biosamples'], row['n_sets'],
                                        row['n_permutation_sets'], row['n_skippable_permutation_sets'], row['n_groups'], row['n_skipped_groups'])
        # get subset of main_df that belongs to this bioproject
        subset_df = main_df[main_df['bio_project'] == bioproject_name]
        n_perm_tests, lambda_jobs = bioproject_obj.batch_lambda_jobs(subset_df, 60)
        total_perm_tests += n_perm_tests
        total_lambda_jobs += lambda_jobs
        # dispatch all jobs for this bioproject
        bioproject_obj.dispatch_all_lambda_jobs(subset_df, LAMBDA_CLIENT)
        bioprojects_dict[row['bioproject']] = bioproject_obj

    log_print(f"Number of bioprojects to process: {num_bioprojects}, Number of ignored bioprojects: {len(bioprojects_list) - num_bioprojects}, "
              f"Number of lambda blocks: {total_lambda_jobs}, Number of permutation tests: {total_perm_tests}")
    # PROGRESS.update_large_progress('Processing', num_bioprojects, 0, 0)

    print("preprocessing time: ", time.time() - prep_start_time)

    # # ================
    # # PROCESSING
    # sync_s3()
    # # ================
    #
    # i = 0
    # while i < num_bioprojects:  # for each block (roughly) in bioprojects
    #     # GET LIST OF BIOPROJECTS IN THIS CURRENT BLOCK
    #     if i + BLOCK_SIZE > num_bioprojects:  # currently inside a non-full block
    #         biopj_block = bioprojects_list[i:]  # just get whatever's left
    #     else:
    #         biopj_block = bioprojects_list[i: i + BLOCK_SIZE]  # get a block's worth of bioprojects
    #
    #     # GET METADATA FOR BIOPROJECTS IN THIS BLOCK
    #     biopj_info_dict, num_skipped = metadata_retrieval(biopj_block, storage)
    #     if len(biopj_info_dict) + num_skipped < BLOCK_SIZE and num_bioprojects > 0:
    #         # handling since we reached the space limit before downloading a full blocks worth of bioprojects
    #         i += len(biopj_info_dict)
    #     else:
    #         i += BLOCK_SIZE  # move to the next block
    #         if len(biopj_info_dict) == 0:
    #             log_print(f"Error: no bioprojects were processed in this block. Moving to the next block.")
    #     num_bioprojects -= num_skipped
    #     del biopj_block
    #
    #     # BEGIN PROCESSING THE BIOPROJECTS IN THIS BLOCK
    #     if not PARALLELIZING:
    #         # PROCESS THE BLOCK
    #         for biopj in biopj_info_dict.keys():
    #             global progress
    #             progress = 0
    #             try:
    #                 output_text_lines = process_bioproject(biopj_info_dict[biopj], main_df)
    #             except Exception as e:
    #                 log_print(f"Error processing bioproject {biopj}: {e}", 2)
    #                 output_text_lines = f'error: {e}'
    #
    #             # OUTPUT FILE (FOR THIS PARTICULAR BIOPROJECT) TODO: postprocessing will involve combining all these files stored on disk
    #             with open(PROBLEMATIC_BIOPJS_FILE, 'a') as f:
    #                 if output_text_lines == 'all groups skipped':
    #                     log_print(f"All groups were skipped for {biopj}. Not creating an output file for it.")
    #                     f.write(f"{biopj} all_groups_skipped\n")
    #                 elif 'error' in output_text_lines:
    #                     log_print(f"Output file for {biopj} was not created due to an error: {output_text_lines}")
    #                     f.write(f"{biopj} output_{output_text_lines}\n")
    #                 elif output_text_lines is not None and output_text_lines:
    #                     try:
    #                         # STORE THE OUTPUT FILE IN temp folder on disk as a file <biopj>_output.csv
    #                         with open(f"{OUTPUT_DIR_DISK}/{biopj}_output_{date}.csv", 'w') as out_file:
    #                             extra_info = 'runtime_seconds,' if PERFOMANCE_STATS else ''
    #                             out_file.write(OUT_COLS_STR % (input_info[0], extra_info) + '\n')
    #                             out_file.write(output_text_lines)
    #                         log_print(f"Output file for {biopj} created successfully")
    #                     except Exception as e:
    #                         log_print(f"Error in creating output file for {biopj} even though we successfully processed it: {e}")
    #                         f.write(f"{biopj} output_error_despite_successful_process\n")
    #                 elif output_text_lines is not None and output_text_lines == '':
    #                     log_print(f"Output file for {biopj} is empty. Not creating a file for it.")
    #                     f.write(f"{biopj} empty_output___this_is_strange\n")
    #                 else:  # output_text_lines is None
    #                     log_print(f"There was a problem with making an output file for {biopj}")
    #                     f.write(f"{biopj} processing_error OR biproject_was_processed_despite_no_associated_runs_provided OR not enough runs provided for this bioproject\n")
    #
    #     else:  # PARALLELIZING
    #         # TODO: IMPLEMENT PARALLELIZING
    #         raise NotImplementedError
    #
    #     # now we're done processing an entire block
    #     # TODO: STORE THE OUTPUT FILES IN S3
    #     # concatenate all the output files in the block into one csv (spans multiple bioprojects)
    #     # and then append-write to output file in the folder <hash_code> (create the output file if it doesn't exist yet)
    #
    #     # FREE UP EVERY ROW IN THE MAIN DATAFRAME THAT BELONGS TO THE CURRENT BLOCK BY INDEXING VIA BIOPROJECT
    #     main_df = main_df[~main_df['bio_project'].isin(biopj_info_dict.keys())]

    # ================
    # POST-PROCESSING
    sync_s3()
    # ================

    # concatenate all the output files in the block into one csv (spans multiple bioprojects)
    if COMBINE_OUTPUTS:
        header_is_written = False
        combined_output_name = f"mwas_output{'_' + date if not S3_OUTPUT_DIR else ''}.csv"
        for file in os.listdir(OUTPUT_DIR_DISK):
            # make sure the file's date is the same as the date mwas started
            try:
                if file.endswith('.csv') and file.split('_output_')[-1].split('.')[0] == date:
                    with open(f"{OUTPUT_DIR_DISK}/{file}", 'r') as f:
                        with open(f"{OUTPUT_DIR_DISK if not S3_OUTPUT_DIR else TEMP_LOCAL_BUCKET}/{combined_output_name}", 'a') as combined:
                            if not header_is_written:
                                extra_info = 'runtime_seconds,' if PERFOMANCE_STATS else ''
                                combined.write(OUT_COLS_STR % (input_info[0], extra_info) + '\n')
                                header_is_written = True
                            next(f)  # ignore first line
                            combined.write(f.read())
                    os.remove(f"{OUTPUT_DIR_DISK}/{file}")
            except Exception as e:  # string splitting error or file not found
                log_print(f"Error while combining output files with: {e}")
                continue


def cleanup() -> Any:
    """Clear out all files in the pickles directory
    """
    if os.path.exists(PICKLE_DIR):
        rmtree(PICKLE_DIR)
        log_print(f"Cleared out all files in {PICKLE_DIR}")

    if os.path.exists('cp_batch_command_list.txt'):
        os.remove('cp_batch_command_list.txt')
    if os.path.exists('ls_batch_command_list.txt'):
        os.remove('ls_batch_command_list.txt')
    if os.path.exists('file_list_with_sizes.txt'):
        os.remove('file_list_with_sizes.txt')

    if isinstance(TEMP_LOCAL_BUCKET, str) and os.path.exists(TEMP_LOCAL_BUCKET):
        # remove the local copy of the s3 bucket
        rmtree(TEMP_LOCAL_BUCKET)


# def display_memory():
#     """Display the current and peak memory usage for debugging purposes"""
#     current, peak = tracemalloc.get_traced_memory()
#     return f"Current memory usage: {current / 10**6} MB; Peak: {peak / 10**6} MB"
#
#
# def top_memory():
#     """Display the top memory usage for debugging purposes"""
#     return [x for x in tracemalloc.take_snapshot().traces._traces if 'mwas_general' in x[2][0][0]]


def main(args: list[str], using_logging=False) -> int | None | tuple[int, str]:
    """Main function to run MWAS on the given data file"""
    num_args = len(args)
    time_start = time.time()
    if using_logging:
        global use_logger
        use_logger = True
    log_print("Starting MWAS (handling arguments)...", 0)

    # Check if the correct number of arguments is provided
    if num_args < 2 or args[1] in ('-h', '--help'):
        log_print("Usage: python mwas_general.py data_file.csv [flags]", 0)
        return 1
    elif args[1].endswith('.csv'):
        global logging_level, IMPLICIT_ZEROS, GROUP_NONZEROS_ACCEPTANCE_THRESHOLD, ALREADY_NORMALIZED, P_VALUE_THRESHOLD, ONLY_T_TEST, \
            COMBINE_OUTPUTS, PERFOMANCE_STATS, S3_OUTPUT_DIR, TEMP_LOCAL_BUCKET, PROBLEMATIC_BIOPJS_FILE
        # s3 storing
        if '--s3-storing' in args:
            try:
                hash_dest = args[args.index('--s3-storing') + 1]
                S3_OUTPUT_DIR = 's3://serratus-biosamples/mwas_data/'  # important to have the trailing slash
                TEMP_LOCAL_BUCKET = f"./{hash_dest}"
                PROBLEMATIC_BIOPJS_FILE = f"{TEMP_LOCAL_BUCKET}/problematic_biopjs.txt"

                # check if the hash_dest exists in our s3 bucket already (if it does, exit mwas)
                process = subprocess.run(SHELL_PREFIX + f"s5cmd ls {S3_OUTPUT_DIR}{hash_dest}/", shell=True, stderr=subprocess.PIPE)
                if not process.stderr:
                    # this implies we found something successfully via ls, so we should exit
                    log_print(f"Warning: {hash_dest} already exists in the s3 bucket. Exiting.", 0)
                    return 0, 'This input has already been processed. Please refer to the s3 bucket for the output.'

                # create local disk folder to sync with s3
                if not os.path.exists(TEMP_LOCAL_BUCKET):
                    os.mkdir(TEMP_LOCAL_BUCKET)

                # create the s3 bucket
                process = subprocess.run(SHELL_PREFIX + f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}/", shell=True)
                if process.returncode == 0:
                    log_print(f"Created s3 bucket: {S3_OUTPUT_DIR}")
                else:
                    log_print(f"Error in creating s3 bucket: {S3_OUTPUT_DIR}", 0)
                    return 1, 'could not create the s3 bucket'

            except Exception as e:
                log_print(f"Error in setting s3 output directory: {e}", 0)
                return 1, 'could not set s3 output directory'
        if '--suppress-logging' in args:
            logging_level = 1
        if '--no-logging' in args:
            logging_level = 0
        if '--explicit-zeros' in args or '--explicit-zeroes' in args:
            IMPLICIT_ZEROS = False
        if '--uncombine-outputs' in args:
            COMBINE_OUTPUTS = False
        if '--t-test-only' in args:
            ONLY_T_TEST = True
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
        if '--performance-stats' in args:
            PERFOMANCE_STATS = True

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

        # CREATE OUTPUT DIRECTORY
        if not os.path.exists(OUTPUT_DIR_DISK):
            os.mkdir(OUTPUT_DIR_DISK)

        register(cleanup)  # handle cleanup on exit

        # RUN MWAS
        log_print("===============\nRunning MWAS...\n===============", 0)
        run_on_file(input_df, (group_by, quantifying_by))
        log_print("MWAS completed successfully", 0)

        # print(display_memory())
        time_taken = round((time.time() - time_start) / 60, 3)
        log_print(f"Time taken: {time_taken} minutes", 0)

        # PROGRESS.update_large_progress('Completed', PROGRESS.num_bioprojects, 0, 0)

        # =================
        # DONE
        sync_s3()
        # =================

        if S3_OUTPUT_DIR:
            # remove the local copy of the s3 bucket
            rmtree(TEMP_LOCAL_BUCKET)
            print(f"Removed local copy of s3 bucket: {TEMP_LOCAL_BUCKET}")

        return 0, f'MWAS completed successfully. Time taken: {time_taken} minutes'
    else:
        log_print("Invalid arguments", 0)
        return 1, 'invalid arguments'


if __name__ == '__main__':
    status = main(sys.argv, False)
    sys.exit(status)
