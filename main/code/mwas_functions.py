"""Generalized MWAS
"""
# built-in libraries
import math
import os
import platform
import pickle
import time
import logging
import json
from typing import Any
from multiprocessing import cpu_count, Process
from threading import Lock

# import required libraries
import boto3
import pandas as pd
import numpy as np
from scipy.stats import permutation_test, ttest_ind_from_stats

# s3 constants
DIR_SUFFIX = '/tmp/'  # '/tmp/' for deployment, '' for local testing
S3_BUCKET_BOTO = 'serratus-biosamples'
S3_METADATA_DIR_BOTO = 'condensed-bioproject-metadata'
S3_OUTPUT_DIR_BOTO = 'mwas-user-dump'
MAIN_DF_PICKLE = 'temp_main_df.pickle'

# file constants
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
BLACKLISTED_METADATA_FIELDS_1 = {
    'publication_date', 'center_name', 'first_public', 'last_public', 'last_update', 'INSDC center name', 'INSDC first public',
    'INSDC last public', 'INSDC last update', 'ENA center name', 'ENA first public', 'ENA last public', 'ENA last update',
    'ENA-FIRST-PUBLIC', 'ENA-LAST-UPDATE', 'DDBJ center name', 'DDBJ first public', 'DDBJ last public', 'DDBJ last update',
    'Contacts/Contact/Name/First', 'Contacts/Contact/Name/Middle', 'Contacts/Contact/Name/Last', 'Contacts/Contact/@email',
    'Name/@url', 'name/@url', 'collected_by', 'when', 'submission_date'
}
BLACKLISTED_METADATA_FIELDS_2 = {
    "anonymized name", "biomaterial_provider", "cell_type", "closest_city", "closest_locality", "collection time", "collection_date",
    "collection_timestamp", "collector", "date collected", "date received", "ena first public", "ena last update", "ena-checklist",
    "ena-first-public", "ena-last-update", "env_broad_scale", "env_local_scale", "env_medium", "event date/time", "event date/time end",
    "event date/time start", "geo_loc_name", "geographic location (depth)", "geographic location (latitude)", "geographic location (region and locality)",
    "insdc center name", "insdc first public", "insdc last update", "last_update", "lat_lon", "latitude", "latitude end", "latitude start",
    "longitude", "longitude end", "longitude start", "name", "name/#text", "package", "pangaea event label", "pangaea sample label",
    "paragraph", "publication_date", "serovar", "specimen collection date", "title"
}
BLACKLISTED_METADATA_FIELDS = BLACKLISTED_METADATA_FIELDS_1.union(BLACKLISTED_METADATA_FIELDS_2)

# constants
NORMALIZING_CONST = 1000000  # 1 million
LAMBDA_RANGE = (128 * 1024 ** 2, 2048 * 1024 ** 2)  # 128 MB to 10240 MB (max lambda memory size
LAMBDA_SIZES = ((2, 900), (4, 5400), (6, 10240))
PERCENT_USED = 0.8  # 80% of the memory will be used
MAX_RAM_SIZE = PERCENT_USED * LAMBDA_RANGE[1]
LAMBDA_CPU_CORES = 6
PROCESS_OVERHEAD_BOUND = 84 * 1024 ** 2  # 84 MB (run module_size_test to determine this, then round up a bit)
OUT_COLS = [
    'bioproject', 'group', 'metadata_field', 'metadata_value', 'test', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
    'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value', 'true_biosamples', 'false_biosamples', 'test_time_duration'
]

logging.basicConfig(level=logging.INFO)


class Config:
    """Class to store configuration settings (flags)"""
    LOGGER = None
    USE_LOGGER = 1
    LOGGING_LEVEL = 2
    IMPLICIT_ZEROS = 1  # if True, then we assume that a missing value is a zero
    MAP_UNKNOWN = 0  # maybe np.nan. Use -1 for when IMPLICIT_ZEROS is False
    GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = 3  # if group has at least this many nonzeros, then it's okay. Note, this flag is only used if IMPLICIT_ZEROS is True
    ALREADY_NORMALIZED = 0  # if it's false, then we need to normalize the data by dividing quantifier by spots
    P_VALUE_THRESHOLD = 0.005
    INCLUDE_SKIPPED_GROUP_STATS = 0  # if True, then the output will include the stats of the skipped tests
    TEST_BLACKLISTED_METADATA_FIELDS = 0  # if False, then the metadata fields in BLACKLISTED_METADATA_FIELDS will be ignored
    PARALLELIZE = 1
    FILE_LOCK = Lock()
    OUT_FILE = None
    TIME_LIMIT = 60
    IGNORE_DYNAMO = 0

    def to_json(self) -> dict:
        """Convert the configuration settings to a dictionary
        note none of the values are boolean originally, so we can convert them to strings"""
        return {
            'IMPLICIT_ZEROS': str(self.IMPLICIT_ZEROS),
            'GROUP_NONZEROS_ACCEPTANCE_THRESHOLD': str(self.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD),
            'ALREADY_NORMALIZED': str(self.ALREADY_NORMALIZED),
            'P_VALUE_THRESHOLD': str(self.P_VALUE_THRESHOLD),
            'INCLUDE_SKIPPED_GROUP_STATS': str(self.INCLUDE_SKIPPED_GROUP_STATS),
            'TEST_BLACKLISTED_METADATA_FIELDS': str(self.TEST_BLACKLISTED_METADATA_FIELDS),
            'LOGGING_LEVEL': str(self.LOGGING_LEVEL),
            'USE_LOGGER': str(self.USE_LOGGER),
            'IGNORE_DYNAMO': str(self.IGNORE_DYNAMO)
        }

    def load_from_json(self, config_dict: dict) -> None:
        """Load the configuration settings from a dictionary"""
        self.IMPLICIT_ZEROS = int(config_dict['IMPLICIT_ZEROS'])
        self.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = int(config_dict['GROUP_NONZEROS_ACCEPTANCE_THRESHOLD'])
        self.ALREADY_NORMALIZED = int(config_dict['ALREADY_NORMALIZED'])
        self.P_VALUE_THRESHOLD = float(config_dict['P_VALUE_THRESHOLD'])
        self.INCLUDE_SKIPPED_GROUP_STATS = int(config_dict['INCLUDE_SKIPPED_GROUP_STATS'])
        self.TEST_BLACKLISTED_METADATA_FIELDS = int(config_dict['TEST_BLACKLISTED_METADATA_FIELDS'])
        self.LOGGING_LEVEL = int(config_dict['LOGGING_LEVEL'])
        self.USE_LOGGER = int(config_dict['USE_LOGGER'])
        if 'TIME_LIMIT' in config_dict:
            self.TIME_LIMIT = float(config_dict['TIME_LIMIT'])
        if 'IGNORE_DYNAMO' in config_dict:
            self.IGNORE_DYNAMO = int(config_dict['IGNORE_DYNAMO'])

    def log_print(self, msg: Any, lvl: int = 1) -> None:
        """Print a message if the logging level is appropriate"""
        if lvl <= self.LOGGING_LEVEL:
            if self.USE_LOGGER and isinstance(self.LOGGER, logging.Logger):
                self.LOGGER.info(msg)
            else:
                print(msg)

    def set_log_level(self, level: int, use: bool) -> None:
        """Set the logging level"""
        Config.LOGGING_LEVEL = level
        if not use:
            Config.USE_LOGGER = 0

    def set_logger(self, file_name: str) -> None:
        """Create a logger"""
        Config.LOGGER = logging.getLogger(file_name)
        if Config.LOGGER.hasHandlers():
            Config.LOGGER.handlers.clear()
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        Config.LOGGER.addHandler(fh)
        Config.LOGGER.setLevel(logging.INFO)


class BioProjectInfo:
    """Class to store information about a bioproject"""

    def __init__(self, config: Config, name: str, metadata_file_size: int, n_biosamples: int,
                 n_sets: int, n_permutation_sets: int, n_skippable_permutation_sets: int,
                 n_groups: int, n_skipped_groups: int, num_lambda_jobs: int = 0, num_conc_procs: int = 0,
                 groups: list[str] = None, jobs: list[tuple[dict[str, tuple[int, int]], int]] = None) -> None:
        self.config = config

        self.name = name
        self.md_file_size = metadata_file_size
        self.n_sets = n_sets
        self.n_permutation_sets = n_permutation_sets
        self.n_skippable_permutation_sets = n_skippable_permutation_sets
        self.n_actual_permutation_sets = n_permutation_sets - n_skippable_permutation_sets \
            if not config.TEST_BLACKLISTED_METADATA_FIELDS else n_permutation_sets
        self.n_biosamples = n_biosamples
        self.n_groups = n_groups
        self.n_skipped_groups = n_skipped_groups
        self.metadata_df = None
        self.metadata_ref_lst = None
        self.metadata_path = None
        self.rpm_map = None
        self.num_lambda_jobs = num_lambda_jobs
        self.lambda_size = 5400  # default lambda size
        self.num_conc_procs = num_conc_procs
        self.groups = [] if groups is None else groups
        self.jobs = [] if jobs is None else jobs  # only used in preprocessing stage

    def to_json(self) -> dict:
        """Convert the bioproject info to a dictionary"""
        return {
            'name': self.name,
            'metadata_file_size': str(self.md_file_size),
            'n_biosamples': str(self.n_biosamples),
            'n_sets': str(self.n_sets),
            'n_permutation_sets': str(self.n_permutation_sets),
            'n_skippable_permutation_sets': str(self.n_skippable_permutation_sets),
            'n_groups': str(self.n_groups),
            'n_skipped_groups': str(self.n_skipped_groups),
            'num_lambda_jobs': str(self.num_lambda_jobs),
            'num_conc_procs': str(self.num_conc_procs),
            'groups': str([str(g) for g in self.groups]) if self.groups else 'everything',
        }

    def batch_lambda_jobs(self, main_df: pd.DataFrame, time_constraint: int) -> tuple[int, int]:
        """
        time_constraint: in seconds
        """
        # choosing lambda size
        mem_width = self.n_biosamples * 240000 + PROCESS_OVERHEAD_BOUND  # this is in bytes
        # note, num of cpu cores = index in LAMBDA_SIZES + 2, so e.g. 2 cores for 900MB, 3 cores for 3600MB, etc.
        size_index = len(LAMBDA_SIZES) - 1  # start with the largest size, since it has better compute power and more cores
        n_conc_procs = LAMBDA_SIZES[size_index][0]
        while mem_width * n_conc_procs * 10.5 < (LAMBDA_SIZES[size_index][1] * 1024 ** 2) * 0.8:
            if size_index == 0:
                break
            size_index -= 1
            n_conc_procs = LAMBDA_SIZES[size_index][0]
        while mem_width * n_conc_procs > (LAMBDA_SIZES[size_index][1] * 1024 ** 2) * PERCENT_USED:
            n_conc_procs -= 1
            if n_conc_procs < 1:
                size_index -= 1
                if size_index == -1:
                    self.config.log_print(f"Error: not enough memory to run a single test on a lambda functions for bio_project")
                    return 0, 0
                n_conc_procs = LAMBDA_SIZES[size_index][0]

        self.num_conc_procs = n_conc_procs
        self.lambda_size = LAMBDA_SIZES[size_index][1]

        time_per_test = self.n_biosamples * 0.0004
        total_tests = (self.n_groups - self.n_skipped_groups) * self.n_actual_permutation_sets
        lambda_area = (math.floor(time_constraint / time_per_test) * n_conc_procs)
        self.num_lambda_jobs = math.ceil(total_tests / lambda_area)

        self.jobs = []
        if self.num_lambda_jobs > 1:
            # remove the groups that have too few nonzeros or are only zeros (leave this in this if block to speed up preprocessing significantly)
            subset_df = main_df[main_df['bio_project'] == self.name]
            group_counts = subset_df['group'].value_counts()
            self.groups = group_counts[group_counts >= self.config.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD].index.tolist()
            for group in subset_df['group'].unique():
                group_subset = subset_df[subset_df['group'] == group]
                all_zeros = not bool(group_subset['quantifier'].sum())
                if all_zeros:
                    self.n_skipped_groups += 1
                    total_tests -= self.n_actual_permutation_sets
                    self.groups.remove(group)
            self.num_lambda_jobs = math.ceil(total_tests / lambda_area)
            if self.num_lambda_jobs < 2:
                self.jobs.append('full')
                return self.n_actual_permutation_sets, self.num_lambda_jobs

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
                        focus_groups[self.groups[curr_group]] = [started_at, left_off_at]
                        curr_group += 1
                        started_at = left_off_at = 0
                if left_off_at > 0:
                    focus_groups[self.groups[curr_group]] = [started_at, left_off_at]
                self.jobs.append(focus_groups)
        elif self.num_lambda_jobs == 1:
            self.jobs.append('full')

        # TODO: estimate time of 'full' jobs, and bunch them together with their t-test lambdas if there's enough time left

        assert self.num_lambda_jobs == len(self.jobs)
        return self.n_actual_permutation_sets, self.num_lambda_jobs

    def dispatch_all_lambda_jobs(self, mwas_id: str, link: str, lam_client: boto3.client, expected_jobs) -> None:
        """lam_client: lambda_client = boto3.client('lambda')

        note lambda handler will call self.process_bioproject(...)
        """
        bioproject_info = self.to_json()
        flags = self.config.to_json()

        self.config.log_print(f"dispatching all {self.num_lambda_jobs} lambda jobs for {self.name}")
        # non permutation test lambda(TODO: lambda[s]?)  always uses the 4 core 5400MB ram lambda
        lam_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps({
                'mwas_id': mwas_id,
                'bioproject_info': bioproject_info,
                'link': link,
                'job_window': 'full',  # empty implies all groups
                'id': 0,  # 0 implies this is the non-perm tests lambda
                'flags': flags,
                'expected_jobs': expected_jobs
            })
        )

        # pick the next lambda size to fit our size
        lambda_alias = f"{self.lambda_size}MBram"
        for i, job in enumerate(self.jobs):  # could be just ['full'] in many cases
            self.config.log_print(f"dispatching lambda job {i + 1} for {self.name}")
            # Invoke the Lambda function
            size_names = {900: "_small", 3600: "", 5400: "", 7200: "_large", 10240: "_large"}
            extra = size_names[self.lambda_size]
            lam_client.invoke(
                FunctionName=f'arn:aws:lambda:us-east-1:797308887321:function:mwas{extra}',
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps({
                    'mwas_id': mwas_id,
                    'bioproject_info': bioproject_info,
                    'link': link,
                    'job_window': job,
                    'id': str(i + 1),
                    'flags': flags,
                    'expected_jobs': expected_jobs
                })
            )

    def retrieve_metadata(self) -> None:
        """load metadata pickle file from s3 into memory as a DataFrame and ref list.
        Note: we should only use this if we're currently working with the bioproject
        """
        try:
            s3 = boto3.client('s3')
            s3.download_file(S3_BUCKET_BOTO, f"{S3_METADATA_DIR_BOTO}/{self.name}.mwaspkl", DIR_SUFFIX + 'temp_file.pickle')
            with open(DIR_SUFFIX + 'temp_file.pickle', 'rb') as f:
                self.metadata_ref_lst = pickle.load(f)
                self.metadata_df = pickle.load(f)
                if len(self.metadata_ref_lst) != self.n_biosamples:
                    self.config.log_print("Warning: metadata ref list and metadata df are not the same length. Requires updating mwas metadata table")
                self.n_biosamples = len(self.metadata_ref_lst)
            os.remove(DIR_SUFFIX + 'temp_file.pickle')
        except Exception as e:
            self.config.log_print(f"Error in loading metadata for {self.name}: {e}", 2)

    def build_rpm_map(self, input_df: pd.DataFrame, groups_focus: dict | None) -> None:
        """Build the rpm map for this bioproject
        """
        if self.rpm_map is not None:  # if the rpm map has already been built
            return
        # rpm map building
        time_build = time.time()
        groups = input_df['group'].unique()
        if not self.config.IMPLICIT_ZEROS:
            self.config.MAP_UNKNOWN = -1  # to allow for user provided 0,
            # we must use a negative number to indicate unknown (user should never provide a negative number)

        # groups_rpm_map is a dictionary with keys as groups and values as a list of two items:
        # the rpm list for the group and a boolean - True => skip the group
        groups_rpm_map = {group: [np.full(self.n_biosamples, self.config.MAP_UNKNOWN, float), False] for group in groups
                          if groups_focus is None or group in groups_focus}
        if self.groups == 'everything':
            self.groups = groups
        # remember, numpy arrays work better with preallocation

        missing_biosamples = set()
        for g in groups_rpm_map.keys():
            group_subset = input_df[input_df['group'] == g]
            if self.config.IMPLICIT_ZEROS:
                if self.config.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD > 0:
                    num_provided = group_subset['quantifier'].count()
                    if num_provided < self.config.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD:
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
                    if self.config.ALREADY_NORMALIZED:
                        groups_rpm_map[g][0][i] = np.mean(reads.values)
                    else:
                        groups_rpm_map[g][0][i] = np.mean([reads.values[x] / (spots.values[x] * NORMALIZING_CONST)
                                                           if spots.values[x] != 0 else self.config.MAP_UNKNOWN
                                                           for x in range(len(biosample_subset))])
                else:
                    if self.config.ALREADY_NORMALIZED:
                        groups_rpm_map[g][0][i] = reads.values[0]
                    else:
                        groups_rpm_map[g][0][i] = reads.values[0] / spots.values[0] * NORMALIZING_CONST \
                            if spots.values[0] != 0 else self.config.MAP_UNKNOWN

        del groups
        if missing_biosamples:  # this is an issue due to the raw csvs not having retrieved certain metadata from ncbi,
            # possibly because the metadata is outdated. Therefore, if this is found, the raw csvs must be remade and then condensed again
            self.config.log_print(
                f"Not found in ref lst: {', '.join(missing_biosamples)} for {self.name} - "
                f"this implies there was no mention of this biosample in the bioproject's metadata file, despite the biosample having "
                f"been provided by the user. Though, this doesn't necessarily mean it's there's no metadata for the biosample on NCBI",
                2)
        self.rpm_map = groups_rpm_map
        self.config.log_print(f"Built rpm map for {self.name} in {round(time.time() - time_build, 2)} seconds.")

    def process_bioproject(self, subset_df: pd.DataFrame, job: dict[str, list[int, int]], id: int) -> bool:
        """Process the given bioproject, and return an output file - concatenation of several group outputs
        """
        time_start = time.time()
        identifier = f"B{id}/{self.num_lambda_jobs}"
        self.config.log_print(f"RETRIEVING METADATA FOR {self.name}...")
        self.retrieve_metadata()
        if self.metadata_df.shape[0] < 1:
            # although no metadata should have 0 rows, since those would be condensed to an empty file
            # and then filtered out in metadata_retrieval, this is just a safety check
            self.config.log_print(f"Skipping {self.name} since its metadata is empty or too small. "
                                  f"(Note: if you see this message, there is a discrepancy in the metadata files)")
            return False
        self.config.log_print(f"BUILDING RPM MAP FOR {self.name}-{identifier}...")
        if self.rpm_map is None:
            if isinstance(job, str) and job == 'full':
                group_focus = None
            else:
                group_focus = job
            self.build_rpm_map(subset_df, group_focus)
        self.config.log_print(f"STARTING TESTS FOR {self.name}-{identifier}...\n")
        if id == 0 or id == '0':  # t-test job
            result = self.process_bioproject_other()
        else:
            if isinstance(job, str):
                result = self.process_bioproject_perms(None, self.n_actual_permutation_sets)
            else:
                num_tests = sum([job[g][1] - job[g][0] for g in job])
                result = self.process_bioproject_perms(job, num_tests)

        self.config.log_print(f"FINISHED PROCESSING {self.name}-{identifier} in {round(time.time() - time_start, 3)} seconds\n")
        return result

    def get_test_stats(self, index_is_include, index_list, rpm_map, group_name, attributes, values) \
            -> tuple[int, int, float, float, float, float, list, list, Any] | None:
        """get the test stats for the given group"""
        # could be optimized?
        true_rpm, false_rpm = [], []
        for i, rpm_val in enumerate(rpm_map):
            if not self.config.IMPLICIT_ZEROS and rpm_val == self.config.MAP_UNKNOWN:
                continue
            if (i in index_list and index_is_include) or (i not in index_list and not index_is_include):
                true_rpm.append(rpm_val)
            else:
                false_rpm.append(rpm_val)

        num_true, num_false = len(true_rpm), len(false_rpm)

        if num_true < 2 or num_false < 2:  # this probably only might happen if IMPLIED_ZEROS is False
            self.config.log_print(f'skipping {group_name} - {attributes}:{values} '
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

    def run_ttests_for_groups(self, groups: list[str]) -> None:
        """function to be run in parallel (groups are run in parallel, within a group is in series)"""
        reusable_results = {}
        results = ''
        for group in groups:
            if max(self.rpm_map[group][0]) == 0:  # then this should have been filtered out in preproc, but probably wasn't because it was in a single lambda job
                self.config.log_print(f"skipping {group} because it has no nonzero reads", 2)
                continue

            for _, row in self.metadata_df.iterrows():
                blacklisted = False
                if not self.config.TEST_BLACKLISTED_METADATA_FIELDS:
                    fields = row['attributes'].split('; ')
                    blacklisted = all(field in BLACKLISTED_METADATA_FIELDS for field in fields)
                    if blacklisted:
                        self.config.log_print(f"skipping blacklisted test {row['attributes']}", 2)
                if row['test_type'] == 't-test' and not (blacklisted or row['skippable?'] == 1):
                    new_row = {
                        'include': row['include?'],
                        'biosample_index_list': row['biosample_index_list'],
                        'group_rpm_list': self.rpm_map[group],
                        'group': group,
                        'attributes': row['attributes'],
                        'values': row['values'],
                        'test_id': -1
                    }
                    result = self.process_set_test(new_row, 't-test', reusable_results)
                    if result is not None:
                        results += result
        self.write_results_to_file(results)

    def process_bioproject_other(self) -> bool:
        """Process the given group, and return an output file
        """
        est_time = (self.n_sets - self.n_actual_permutation_sets) * 0.0001 * (self.n_groups - self.n_skipped_groups)  # TODO: replace 0.0001 with actual t-test time constant
        try:
            if est_time <= self.config.TIME_LIMIT:
                self.run_ttests_for_groups(self.groups)
            else:
                num_workers = cpu_count()
                workers = {}
                num_instances = len(self.groups)
                instance_range = num_instances // num_workers
                for i in range(num_workers):
                    batch = self.groups[i * instance_range: min((i + 1) * instance_range, num_instances)]
                    workers[i] = Process(target=self.run_ttests_for_groups, args=(batch,))
                    workers[i].start()
                for i in range(num_workers):
                    workers[i].join()
            self.config.log_print(f"Finished processing tests for {self.name}\n")
        except Exception as e:
            self.config.log_print(f"Error in processing bioproject {self.name}: {e}", 2)
            return False
        self.config.log_print(f"estimated time: {est_time}, time limit: {self.config.TIME_LIMIT}")
        return True

    def write_results_to_file(self, results: str) -> None:
        """Write the results to a file"""
        file_lock = self.config.FILE_LOCK
        with file_lock:
            with open(self.config.OUT_FILE, 'a') as f:
                f.write(results)

    def permu_batch(self, tests):
        """batches to be parallelized for permutation test mode
        """
        results_str = ''
        reusable_keys = {}
        for test in tests:
            result = self.process_set_test(test, 'permutation-test', reusable_keys)
            if result is not None:
                results_str += result
        self.write_results_to_file(results_str)

    def process_bioproject_perms(self, job_groups: dict | None, num_tests: int) -> bool:
        """Process the given bioproject, and return an output file - concatenation of several group outputs
        """
        self.config.log_print(f"Performing {num_tests} permutation tests for {self.name}...")
        tests = []
        test_id = 0
        if job_groups is not None:
            groups = job_groups.keys()
        else:
            groups = self.groups  # note: was set from 'everything' to actual list in build_rpm
        # filter the metadata_df to only include the permutation tests and the non-skippable tests
        metadata_df = self.metadata_df[self.metadata_df['test_type'] == 'permutation-test']
        if not self.config.TEST_BLACKLISTED_METADATA_FIELDS:
            metadata_df = metadata_df[metadata_df['skippable?'] == 0]

        for group in groups:
            if max(self.rpm_map[group][0]) == 0:  # then this should have been filtered out in preproc, but probably wasn't because it was in a single lambda job
                self.config.log_print(f"skipping {group} because it has no nonzero reads", 2)
                continue
            if job_groups is not None:
                start, end = job_groups[group]
                sets_subset_df = metadata_df.iloc[
                                 start:end]  # note: this is correct. Checking correctness via the metadata csv will show it's off by 2 - that's due to indexing from 1 in csv, and counting the header
            else:
                sets_subset_df = metadata_df

            # add the tests to the tests list
            for _, row in sets_subset_df.iterrows():
                blacklisted = False
                if not self.config.TEST_BLACKLISTED_METADATA_FIELDS:
                    fields = row['attributes'].split('; ')
                    blacklisted = all(field in BLACKLISTED_METADATA_FIELDS for field in fields)
                if row['test_type'] == 't-test' or blacklisted or row['skippable?'] == 1:
                    self.config.log_print(f"skipping blacklisted test {row['attributes']}", 2)
                    continue
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
        num_tests = len(tests)
        try:
            num_workers = min(cpu_count(), self.num_conc_procs)
            series_time = 0.0004 * self.n_biosamples * num_tests
            if self.config.PARALLELIZE and num_workers > 1 and series_time > self.config.TIME_LIMIT / 4:
                workers = {}
                num_instances = len(tests)
                instance_range = num_instances // num_workers
                for i in range(num_workers):
                    batch = tests[i * instance_range: min((i + 1) * instance_range, num_instances)]
                    workers[i] = Process(target=self.permu_batch, args=(batch,))
                    workers[i].start()
                for i in range(num_workers):
                    workers[i].join()
            else:
                self.config.log_print(f"running tests in series {'because it is small enough for time limit' if series_time >= self.config.TIME_LIMIT else ''}")
                self.permu_batch(tests)
            self.config.log_print(f"Finished processing tests for {self.name}\n")
            self.config.log_print(f"estimated series time: {series_time}, time limit: {self.config.TIME_LIMIT}")
        except Exception as e:
            self.config.log_print(f"Error in processing bioproject {self.name}: {e}", 2)
            return False
        return True

    def process_set_test(self, row: dict, test_type: str, reusable_results: dict) -> None | str:
        """Process a single set of permutation tests
        if test_type is t_test then results is a string,
        if test_type is permutation_test then results is a dictionary
        """
        test_start_time = time.time()

        group, rpm_list = row['group'], row['group_rpm_list'][0]
        index_is_inlcude, index_list = row['include'], row['biosample_index_list']
        test_info = self.get_test_stats(index_is_inlcude, index_list, rpm_list, group, row['attributes'], row['values'])
        if test_info is None:
            return
        num_true, num_false, mean_rpm_true, mean_rpm_false, sd_rpm_true, sd_rpm_false, true_rpm, false_rpm, fold_change = test_info
        test_key = num_true, num_false, mean_rpm_true, mean_rpm_false, sd_rpm_true, sd_rpm_false, fold_change, group
        if test_key in reusable_results and (mean_rpm_false == 0 or mean_rpm_true == 0):
            test_statistic, p_value, status = reusable_results[test_key]
            self.config.log_print(f"Reusing results for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}", 2)
        else:
            self.config.log_print(f"Running {test_type} for bioproject: {self.name}, group: {group}, set: {row['attributes']}:{row['values']}", 2)
            try:
                if test_type == 't-test':
                    # T TEST
                    assert min(num_false, num_true) < 4
                    status = 't-test'
                    test_statistic, p_value = ttest_ind_from_stats(mean1=mean_rpm_true, std1=sd_rpm_true, nobs1=num_true, mean2=mean_rpm_false, std2=sd_rpm_false, nobs2=num_false,
                                                                   equal_var=False)
                elif test_type == 'permutation-test':
                    # PERMUTATION TEST
                    status = 'permutation-test'
                    num_samples = 10000  # note, we do not need to lower this to be precise (using n choose k) since scipy does this for us anyway

                    def mean_diff_statistic(x, y, axis):
                        """if -ve, then y is larger than x"""
                        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

                    res = permutation_test((true_rpm, false_rpm), statistic=mean_diff_statistic, n_resamples=num_samples, vectorized=True)
                    p_value, test_statistic = res.pvalue, res.statistic

                else:
                    self.config.log_print(f"Error: unknown test type {test_type}", 2)
                    return
                reusable_results[test_key] = (test_statistic, p_value, status)
            except Exception as e:
                self.config.log_print(f"Error running permutation test for {row['group']} - {row['attributes']}:{row['values']} - {e}", 2)
                return

        if p_value < self.config.P_VALUE_THRESHOLD:
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
        self.config.log_print(f"Finished with p-value: {p_value}", 2)

        test_end_time = time.time() - test_start_time
        self.config.log_print(f"Test took {test_end_time} seconds", 2)

        # record the output
        if test_statistic == "-inf":
            test_statistic = "negative inf"
        this_result = (
            f"{self.name},{row['group']},{row['attributes'].replace(',', ' ')},{row['values'].replace(',', ' ')},{status},"
            f"{num_true},{num_false},{mean_rpm_true},{mean_rpm_false},{sd_rpm_true},{sd_rpm_false},{fold_change},"
            f"{test_statistic},{p_value},{true_biosamples},{false_biosamples},{test_end_time}\n"
        )
        self.config.log_print(this_result, 2)
        return this_result


def load_bioproject_from_dict(config: Config, bioproject_dict: dict) -> BioProjectInfo:
    """Load a bioproject from a dictionary"""
    groups_str = bioproject_dict['groups']
    return BioProjectInfo(
        config,
        bioproject_dict['name'],
        int(bioproject_dict['metadata_file_size']),
        int(bioproject_dict['n_biosamples']),
        int(bioproject_dict['n_sets']),
        int(bioproject_dict['n_permutation_sets']),
        int(bioproject_dict['n_skippable_permutation_sets']),
        int(bioproject_dict['n_groups']),
        int(bioproject_dict['n_skipped_groups']),
        int(bioproject_dict['num_lambda_jobs']),
        int(bioproject_dict['num_conc_procs']),
        groups_str if not groups_str[0] == '[' and not groups_str[-1] == ']' else eval(groups_str)
    )


def lambda_job_json(bioproject: BioProjectInfo, s3_dir: str, ttest_job: bool):
    """only intended for testing purposes (get event without actiavting lambdas"""
    bioproject_info = bioproject.to_json()
    flags = bioproject.config.to_json()
    return {
        'bioproject_info': bioproject_info,
        'link': s3_dir,
        'job_window': 'full' if ttest_job else bioproject.jobs[0],
        'id': 0 if ttest_job else 1,
        'flags': flags
    }
