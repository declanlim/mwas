"""Generalized MWAS
"""
# built-in libraries
import math
import platform
import logging
import json
from typing import Any
from threading import Lock

# import required libraries
import boto3
import pandas as pd

# s3 constants
DIR_SUFFIX = '/tmp/'  # '/tmp/' for deployment, '' for local testing
S3_BUCKET_BOTO = 'serratus-biosamples'
S3_METADATA_DIR_BOTO = 'condensed-bioproject-metadata'
S3_OUTPUT_DIR_BOTO = 'mwas_data'
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
BLACKLISTED_METADATA_FIELDS = {
    'publication_date', 'center_name', 'first_public', 'last_public', 'last_update', 'INSDC center name', 'INSDC first public',
    'INSDC last public', 'INSDC last update', 'ENA center name', 'ENA first public', 'ENA last public', 'ENA last update',
    'ENA-FIRST-PUBLIC', 'ENA-LAST-UPDATE', 'DDBJ center name', 'DDBJ first public', 'DDBJ last public', 'DDBJ last update',
    'Contacts/Contact/Name/First', 'Contacts/Contact/Name/Middle', 'Contacts/Contact/Name/Last', 'Contacts/Contact/@email',
    'Name/@url', 'name/@url', 'collected_by', 'when', 'submission_date'
}

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
            'USE_LOGGER': str(self.USE_LOGGER)
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
