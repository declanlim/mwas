"""main function to run MWAS on the given data file, i.e. starting point"""
import sys
import psycopg2
from shutil import rmtree
from mwas_functions import *
import subprocess

# globals
DATE = time.asctime().replace(' ', '_').replace(':', '-')
TEMP_LOCAL_BUCKET = None  # tbd when hash code is provided
S3_METADATA_DIR = 's3://serratus-biosamples/condensed-bioproject-metadata'
S3_OUTPUT_DIR = 's3://serratus-biosamples/mwas_data/'
CONFIG = Config()

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
    SELECT * FROM mwas_mdfiles_info
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


def update_progress(status, num_bioprojects, num_lambda_jobs, num_permutation_tests, num_jobs_completed, start_time, sync_s3=True) -> None:
    """Update the progress file with the given progress dictionary"""
    with open(f"{TEMP_LOCAL_BUCKET}/{PROGRESS_FILE}", 'w') as f:
        json.dump({
            'status': status,
            'num_bioprojects': str(num_bioprojects),
            'num_lambda_jobs': str(num_lambda_jobs),
            'num_permutation_tests': str(num_permutation_tests),
            'num_jobs_completed': str(num_jobs_completed),
            'time_elapsed': str(time.time() - start_time),
            'start_time': str(start_time)
        }, f)
    if sync_s3:
        s3_sync()


def s3_sync():
    """Sync the local s3 bucket with the s3 bucket"""
    process = subprocess.run(SHELL_PREFIX + f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}", shell=True)
    if process.returncode == 0:
        CONFIG.log_print(f"Synced s3 bucket: {S3_OUTPUT_DIR}")
    else:
        CONFIG.log_print(f"Error in syncing s3 bucket: {S3_OUTPUT_DIR}", 0)


def concat_s3_results(s3_dir: str) -> None:
    """Concatenate all the results in the s3 bucket outputs directory into a single csv file

    NOTE: this should be moved to a new lambda, and shouldn't be in preproc code (this file)
    """
    # create a local dir to store the results
    results_dir = f"{TEMP_LOCAL_BUCKET}/{OUTPUT_FILES_DIR}"
    os.mkdir(results_dir)

    process = subprocess.run(SHELL_PREFIX + f"s5cmd cp -r {S3_OUTPUT_DIR} {results_dir}", shell=True)
    if process.returncode == 0:
        CONFIG.log_print(f"Successfully copied s3 bucket: {S3_OUTPUT_DIR}")
    else:
        CONFIG.log_print(f"Error in copying s3 bucket: {S3_OUTPUT_DIR}", 0)
        return

    # concatenate all the result files into a single csv file
    with open(f"{TEMP_LOCAL_BUCKET}/{OUTPUT_CSV_FILE}", 'w') as final_output:
        final_output.write(','.join(OUT_COLS) + ',job_id\n')  # write the header
        for file in os.listdir(results_dir):  # for each file in the results directory
            if file.endswith('.txt'):
                with open(f"{results_dir}/{file}", 'r') as result_file:
                    # get lines from the file
                    lines = result_file.readlines()
                    for line in lines:
                        # filter blank lines
                        if ',' in line:  # lines with commas are the ones we want since they have csv data
                            final_output.write(line)
    rmtree(results_dir)
    CONFIG.log_print(f"Concatenated all results into {OUTPUT_CSV_FILE}")


def get_table_via_sql_query(connection: dict, query: str, accs: list[str]) -> pd.DataFrame | None:
    """Get the data from the given table using the given query with the given accessions
    """
    try:
        conn = psycopg2.connect(**connection)
        conn.close()  # close the connection because we are only checking if we can connect
        CONFIG.log_print(f"Successfully connected to database at {connection['host']}")
    except psycopg2.Error:
        CONFIG.log_print(f"Unable to connect to database at {connection['host']}")

    accs_str = ", ".join([f"'{acc}'" for acc in accs])

    with psycopg2.connect(**connection) as conn:
        try:
            df = pd.read_sql(query % accs_str, conn)
            return df
        except psycopg2.Error:
            CONFIG.log_print("Error in executing query")
            return None


def preprocessing(data_file: pd.DataFrame, start_time: float, hash_dest: str) -> tuple[int, int, int] | None:
    """Run MWAS on the given data file
    input_info is a tuple of the group and quantifier column names
    storage will usually be the tmpfs mount point mount_tmpfs
    """
    problematic_biopjs_file_actual = f"{TEMP_LOCAL_BUCKET}/{PROBLEMATIC_BIOPJS_FILE}"
    CONFIG.log_print(f"Starting MWAS at {DATE}")

    # ================
    # PRE-PROCESSING
    # ================
    prep_start_time = time.time()

    update_progress('Preprocessing', 0, 0, 0, 0, start_time)

    # CREATE BIOSAMPLES DATAFRAME
    runs = data_file['run'].unique()
    main_df = get_table_via_sql_query(LOGAN_CONNECTION_INFO,
                                      LOGAN_SRA_QUERY[0] if not CONFIG.ALREADY_NORMALIZED else LOGAN_SRA_QUERY[1], runs)
    if main_df is None:
        return
    if not CONFIG.ALREADY_NORMALIZED:
        main_df['spots'] = main_df['spots'].replace(0, NORMALIZING_CONST)

    # merge the main_df with the input_df (must assume both have run column)
    main_df = main_df.merge(data_file, on='run', how='outer')
    main_df.fillna(CONFIG.MAP_UNKNOWN, inplace=True)
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
        num_skipped_groups = (df['group'].value_counts() < int(CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD)).sum()
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
        bioproject_obj = BioProjectInfo(CONFIG, bioproject_name, row_srs['condensed_md_file_size'].iloc[0], row_srs['n_biosamples'].iloc[0], row_srs['n_sets'].iloc[0],
                                        row_srs['n_permutation_sets'].iloc[0], row_srs['n_skippable_permutation_sets'].iloc[0],
                                        row_srs['n_groups'].iloc[0], row_srs['n_skipped_groups'].iloc[0])

        n_perm_tests, lambda_jobs = bioproject_obj.batch_lambda_jobs(main_df, 60)
        total_perm_tests += n_perm_tests
        total_lambda_jobs += lambda_jobs + 1  # +1 for the non-perm test lambda
        bioprojects_dict[bioproject_name] = bioproject_obj
        CONFIG.log_print(f"{bioproject_name} had {lambda_jobs} lambda jobs and {n_perm_tests} permutation tests")

    CONFIG.log_print(f"Number of bioprojects to process: {num_bioprojects}, Number of ignored bioprojects: {len(bioprojects_list) - num_bioprojects}, "
                     f"Number of lambda jobs: {total_lambda_jobs}, Number of permutation tests: {total_perm_tests}")

    CONFIG.log_print(f"preprocessing time: {time.time() - prep_start_time}", 1)

    # pickle the main_df
    main_df.to_pickle(f'{TEMP_LOCAL_BUCKET}/{MAIN_DF_PICKLE}')

    # this will sync the main_df file to the s3 dir
    update_progress('Dispatching Lambdas', num_bioprojects, total_lambda_jobs, total_perm_tests, 0, start_time)
    del biopj_info_df, main_df

    # graph this to see lambda job count distribution: sorted([bioprojects_dict[x].num_lambda_jobs for x in bioprojects_dict])

    LAMBDA_CLIENT = boto3.client('lambda')
    # TODO: invoke postprocessing lambda that also acts as a subscriber to the sns topic, so it recieves lambda notifications and then begins postprocessing once all lambdas are done






    for bioproject in bioprojects_dict:
        bioproject_obj = bioprojects_dict[bioproject]
        bioproject_obj.dispatch_all_lambda_jobs(hash_dest, LAMBDA_CLIENT)
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
            global TEMP_LOCAL_BUCKET

            if '--s3' not in args:
                hash_dest = DATE
            else:
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
            CONFIG.set_logger(preproc_log_file_actual)
            CONFIG.set_log_level(0 if '--no-logging' in args else (1 if '--suppress-logging' in args else 2),
                                 False if '--print-mode' in args else True)
            with open(problematic_biopjs_file_actual, 'w') as f:
                f.write('bioproject,reason\n')
            update_progress('Starting', 0, 0, 0, 0, time_start, False)

            # create the s3 bucket
            process = subprocess.run(SHELL_PREFIX + f"s5cmd sync {TEMP_LOCAL_BUCKET} {S3_OUTPUT_DIR}", shell=True)
            if process.returncode == 0:
                CONFIG.log_print(f"Created s3 bucket: {S3_OUTPUT_DIR}")
            else:
                CONFIG.log_print(f"Error in creating s3 bucket: {S3_OUTPUT_DIR}", 0)
                return 1, 'could not create the s3 bucket'
        except Exception as e:
            CONFIG.log_print(f"Error in setting s3 output directory: {e}", 0)
            return 1, 'could not set s3 output directory'

        # set flags

        if '--already-normalized' in args:  # TODO: test
            CONFIG.ALREADY_NORMALIZED = 1
        if '--p-value-threshold' in args:
            try:
                CONFIG.P_VALUE_THRESHOLD = float(args[args.index('--p-value-threshold') + 1])
            except Exception as e:
                CONFIG.log_print(f"Error in setting p-value threshold: {e}", 0)
                return 1
        if '--group-nonzero-threshold' in args:
            try:
                CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = int(args[args.index('--group-nonzero-threshold') + 1])
            except Exception as e:
                CONFIG.log_print(f"Error in setting group nonzeros threshold: {e}", 0)
                return 1
        if '--explicit-zeros' in args or '--explicit-zeroes' in args:
            CONFIG.IMPLICIT_ZEROS = 0
            if CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD < 4:
                CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = 4

        try:  # reading the input file
            input_df = pd.read_csv(args[1])  # arg1 is a file path
            # rename group and quantifier columns to standard names, and also save original names
            group_by, quantifying_by = input_df.columns[1], input_df.columns[2]
            input_df.rename(columns={
                input_df.columns[0]: 'run', input_df.columns[1]: 'group', input_df.columns[2]: 'quantifier'
            }, inplace=True)

            # assume it has three columns: run, group, quantifier. And the group and quantifier columns have special names
            if len(input_df.columns) != 3:
                CONFIG.log_print("Data file must have three columns in this order: <run>, <group>, <quantifier>", 0)
                return 1, 'invalid data file'

            # attempt to correct column types so the next if block doesn't exit us
            input_df['run'] = input_df['run'].astype(str)
            input_df['group'] = input_df['group'].astype(str)
            input_df['quantifier'] = pd.to_numeric(input_df['quantifier'], errors='coerce')

            # check if run and group contain string values and quantifier contains numeric values
            if (input_df['run'].dtype != 'object' or input_df['group'].dtype != 'object'
                    or input_df['quantifier'].dtype not in ('float64', 'int64')):
                CONFIG.log_print("run and group column must contain string values, and quantifier column must contain numeric values", 0)
                return 1
        except FileNotFoundError:
            CONFIG.log_print("File not found", 0)
            return 1, 'file not found'

        # RUN MWAS
        CONFIG.log_print("RUNNING MWAS...", 0)
        num_bioprojects, num_lambda_jobs, num_permutation_tests = preprocessing(input_df, time_start, hash_dest)
        CONFIG.log_print(f"Time taken: {time.time() - time_start} minutes", 0)

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
        # rmtree(TEMP_LOCAL_BUCKET)  problem: it tries removing the log file, which it cannot do while using it
        print(f"Removed local copy of s3 bucket: {TEMP_LOCAL_BUCKET}")

        return 0, f'Preprocessing completed in {time.time() - time_start} seconds'
    else:
        CONFIG.log_print("Invalid arguments", 0)
        return 1, 'invalid arguments'


if __name__ == '__main__':
    exit_code = main(sys.argv)
    sys.exit(exit_code)