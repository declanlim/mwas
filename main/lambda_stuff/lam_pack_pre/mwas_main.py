"""main function to run MWAS on the given data file, i.e. starting point"""
import os
import time
import uuid
import psycopg2
from mwas_functions_for_preprocessing import *

# globals
DATE = time.asctime().replace(' ', '_').replace(':', '-')
TEMP_LOCAL_BUCKET = None  # tbd when hash code is provided
S3_MWAS_DATA = 'mwas-user-dump'
HASH_LINK = "default"
CONFIG = Config()

s3_client = boto3.client('s3')

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
            'start_time': str(start_time),
            'preprocessing_time': str(time.time() - start_time)
        }, f)
    if sync_s3:
        s3_sync()


def s3_sync():
    """Sync the local s3 bucket with the s3 bucket"""
    try:
        file_list = os.listdir(TEMP_LOCAL_BUCKET)
        CONFIG.log_print(file_list)
        for file in file_list:
            if '.' in file:  # file
                s3_client.upload_file(f"{TEMP_LOCAL_BUCKET}/{file}", S3_MWAS_DATA, f"{HASH_LINK}/{file}")
                # CONFIG.log_print(f"Uploaded {file} to s3 bucket: {S3_MWAS_DATA}/{HASH_LINK}")
            else:  # folder
                s3_path = f"{HASH_LINK}/{file}/"
                s3_client.put_object(Bucket=S3_MWAS_DATA, Key=s3_path)
                # CONFIG.log_print(f"made s3 folder: {S3_MWAS_DATA}/{s3_path}")
        CONFIG.log_print(f"Synced s3 bucket: {S3_MWAS_DATA}/{HASH_LINK}")
    except Exception as e:
        CONFIG.log_print(f"Error in syncing s3 bucket: {S3_MWAS_DATA}/{HASH_LINK}: {e}", 0)


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


def preprocessing(data_file: pd.DataFrame, start_time: float) -> tuple[int, int, int] | None:
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

    mwas_id = str(uuid.uuid4())
    # DISPATCH ALL LAMBDA JOBS
    # LAMBDA_CLIENT = boto3.client('lambda')
    # job_list = []
    # flags = CONFIG.to_json()
    # for bioproject in bioprojects_dict:
    #     bioproject_obj = bioprojects_dict[bioproject]
    #     bioproject_obj.get_jobs(job_list)
    # # event limit is 256KB, one job message is about 390bytes, round to 400. then add 500 to account for other data
    # # so we'll do it in batches of 600 (which is roughly min(1200, (1024 * 256 - 500) / 400))
    # batch_size, left_off = 600, 0
    # job_is_done = False
    # while not job_is_done:
    #     for i in range(left_off, len(job_list), batch_size):
    #         if len(job_list) - i < batch_size:
    #             job_slice = job_list[i:]
    #             CONFIG.log_print(f"Dispatching jobs {i} to {len(job_list)}")
    #         else:
    #             job_slice = job_list[i:i + batch_size]
    #             CONFIG.log_print(f"Dispatching jobs {i} to {i + batch_size}")
    #         payload = {
    #             'jobs': job_slice,
    #             'mwas_id': mwas_id,
    #             'flags': flags,
    #             'link': HASH_LINK,
    #             'expected_jobs': total_lambda_jobs
    #         }
    #         try:
    #             LAMBDA_CLIENT.invoke(
    #                 FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas_pre_helper',
    #                 InvocationType='Event',  # Asynchronous invocation
    #                 Payload=json.dumps(payload)
    #             )
    #         except Exception as e:
    #             CONFIG.log_print(f"Error in invoking lambda: {e}", 0)
    #             left_off = i
    #             batch_size = max(10, batch_size // 2)
    #             break
    #     job_is_done = True

    SNS_CLIENT = boto3.client('sns')
    flags = CONFIG.to_json()
    for bioproject in bioprojects_dict:
        bioproject_obj = bioprojects_dict[bioproject]
        bioproject_obj.dispatch_all_lambda_jobs(mwas_id, HASH_LINK, SNS_CLIENT, total_lambda_jobs, flags, direct=False)
        del bioproject_obj

    return num_bioprojects, total_lambda_jobs, total_perm_tests


def main(flags: dict):
    """Main function to run MWAS on the given data file"""
    time_start = time.time()

    try:
        global TEMP_LOCAL_BUCKET

        # create local disk folder to sync with s3
        uid = str(uuid.uuid4())
        TEMP_LOCAL_BUCKET = f"{DIR_SUFFIX}{uid}"
        os.mkdir(TEMP_LOCAL_BUCKET)
        problematic_biopjs_file_actual = f"{TEMP_LOCAL_BUCKET}/{PROBLEMATIC_BIOPJS_FILE}"
        preproc_log_file_actual = f"{TEMP_LOCAL_BUCKET}/{PREPROC_LOG_FILE}"
        # make subfolders for outputs and logs
        os.mkdir(f"{TEMP_LOCAL_BUCKET}/{OUTPUT_FILES_DIR}")
        os.mkdir(f"{TEMP_LOCAL_BUCKET}/{PROC_LOG_FILES_DIR}")

        # we don't care if the hash dest already exists. we assume this was checked in presigned url lambda. So we don't worry here
        # get csv from the hash dest
        try:
            s3_client.download_file(S3_MWAS_DATA, f"{HASH_LINK}/input.csv", f"{TEMP_LOCAL_BUCKET}/input.csv")
        except Exception as e:
            CONFIG.log_print(f"Error in downloading input file: {e}", 0)
            return 1, 'could not download the input file'
        # scan TEMP_LOCAL_BUCKET for file
        actual_input_file = ""
        for file in os.listdir(TEMP_LOCAL_BUCKET):
            if 'input.csv' in file:
                actual_input_file = file
        if not actual_input_file:
            CONFIG.log_print(f"Error in downloading input file: there was no input.csv found...", 0)
            return 1, 'file not found'

        # create the problematic bioprojects file and logger and progress.json
        CONFIG.set_logger(preproc_log_file_actual)
        CONFIG.set_log_level(0 if 'no_logging' in flags else (1 if 'suppress_logging' in flags else 2),
                             False if 'print_mode' in flags else True)
        with open(problematic_biopjs_file_actual, 'w') as f:
            f.write('bioproject,reason\n')
        update_progress('Starting', 0, 0, 0, 0, time_start, False)

        # assume the s3 bucket exists, otherwise we wouldn't have reached this line

    except Exception as e:
        CONFIG.log_print(f"Error in setting s3 output directory: {e}", 0)
        return 1, 'could not set s3 output directory'

    # set flags
    if 'ignore_dynamo' in flags:
        CONFIG.IGNORE_DYNAMO = 1
    if 'already_normalized' in flags:  # TODO: test
        CONFIG.ALREADY_NORMALIZED = flags['already_normalized']
    if 'p_value_threshold' in flags:
        try:
            CONFIG.P_VALUE_THRESHOLD = float(flags['p_value_threshold'])
        except Exception as e:
            CONFIG.log_print(f"Error in setting p-value threshold: {e}", 0)
            return 1
    if 'group_nonzero_threshold' in flags:
        try:
            CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = int(flags['group_nonzero_threshold'])
        except Exception as e:
            CONFIG.log_print(f"Error in setting group nonzeros threshold: {e}", 0)
            return 1
    if 'explicit_zeros' in flags or 'explicit_zeroes' in flags:
        CONFIG.IMPLICIT_ZEROS = 0
        if CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD < 4:
            CONFIG.GROUP_NONZEROS_ACCEPTANCE_THRESHOLD = 4

    try:  # reading the input file
        input_df = pd.read_csv(f"{TEMP_LOCAL_BUCKET}/{actual_input_file}")
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
    num_bioprojects, num_lambda_jobs, num_permutation_tests = preprocessing(input_df, time_start)
    CONFIG.log_print(f"Time taken: {time.time() - time_start} seconds", 0)

    update_progress('Processing', num_bioprojects, num_lambda_jobs, num_permutation_tests, 0, time_start)

    # remove the local copy of the s3 bucket
    # rmtree(TEMP_LOCAL_BUCKET)  problem: it tries removing the log file, which it cannot do while using it
    # print(f"Removed local copy of s3 bucket: {TEMP_LOCAL_BUCKET}")

    return 0, f'Preprocessing completed in {time.time() - time_start} seconds'


def lambda_handler(event, context):
    """Main function to run MWAS on the given data file.
    should have flags dict and hash"""
    print(context)
    print(event)

    try:
        if 'body' not in event:
            body = event
        else:
            body = event['body']
            if isinstance(body, str):  # addresses a string key issue
                body = json.loads(body)

        global HASH_LINK
        HASH_LINK = body['hash']
        flags = body['flags']
    except KeyError:
        return {
            'statusCode': 400,
            'message': 'hash and flags must be provided in the event body'
        }
    ret = main(flags)
    print(f"finished with: {ret}")
    return {
        'statusCode': 200 if ret[0] == 0 else 500,
        'message': ret[1]
    }
