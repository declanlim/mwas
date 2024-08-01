"""Lambda function - for statistical tests - for MWAS"""

from mwas_functions import *


def lambda_handler(event: dict, context) -> dict:
    """Lambda handler for MWAS"""
    # get the data from event
    bioproject_info = event['bioproject_info']
    main_df_link = event['main_df_link']
    job_window = event['job_window']
    lam_id = event['id']

    CONFIG = Config()
    CONFIG.load_from_json(event['flags'])
    CONFIG.set_logger(f"log_{bioproject_info['name']}_job{lam_id}.txt")

    CONFIG.log_print(f"Starting MWAS processing for {bioproject_info['name']} job {lam_id}.", 1)
    CONFIG.log_print(f"Lambda info: {context}", 1)

    # get the main_df
    try:
        command_cp = f"s5cmd cp -f {main_df_link} main_df.pickle"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        with open('main_df.pickle', 'rb') as f:
            main_df = pickle.load(f)
        os.remove('main_df.pickle')

    except Exception as e:
        CONFIG.log_print(f"Error in loading main_df: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in loading main_df: {e}"}

    # TODO: for loop here to handle multiple bioprojects in series if the event says so. This will be to optimize for smaller bioprojects, so we don't need to make a new lambda
    #  for each one - which would be especially costly when main_df is large get the bioproject info
    bioproject = load_bioproject_from_dict(CONFIG, bioproject_info)
    # get subset of main_df
    subset_df = main_df[main_df['bio_project'] == bioproject.name]
    del main_df

    result = bioproject.process_bioproject(subset_df, job_window, lam_id)  # it needs to be the subset of main_df
    if result is None:
        return {'statusCode': 500, 'body': f"Error in processing bioproject {bioproject.name} job {lam_id}"}
    # make result file
    with open('result.txt', 'w') as f:
        f.write(result)

    # upload the result and log files to s3
    try:
        command_cp = f"s5cmd cp -f result.txt {S3_OUTPUT_DIR}/{OUTPUT_FILES_DIR}/result_{bioproject.name}_job{lam_id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove('result.txt')

        command_cp = f"s5cmd cp -f log_{bioproject.name}_job{lam_id}.txt {S3_OUTPUT_DIR}/{PROC_LOG_FILES_DIR}/log_{bioproject.name}_job{lam_id}.txt"
        subprocess.run(SHELL_PREFIX + command_cp, shell=True)
        os.remove(f"log_{bioproject.name}_job{lam_id}.txt")

    except Exception as e:
        CONFIG.log_print(f"Error in syncing output: {e}", 2)
        return {'statusCode': 500, 'body': f"Error in putting output to s3: {e}"}

    return {'statusCode': 200, 'body': f'MWAS processing completed for {bioproject.name} job {lam_id}'}


if __name__ == '__main__':
    # test
    event = {'bioproject_info': {'name': 'PRJDB7993',
                                 'metadata_file_size': '132476',
                                 'n_biosamples': '1990',
                                 'n_sets': '1079',
                                 'n_permutation_sets': '388',
                                 'n_skippable_permutation_sets': '0',
                                 'n_groups': '20',
                                 'n_skipped_groups': '0',
                                 'num_lambda_jobs': '26',
                                 'num_conc_procs': '3',
                                 'groups': "['IFNA1', 'IFNA2', 'IFNL3', 'IFNL2', 'IFNL1', 'IFNG', 'IFNW1', 'IFNB1', 'IFNA21', 'IFNA17', 'IFNA16', 'IFNA14', 'IFNA13', 'IFNA10', "
                                           "'IFNA8', 'IFNA7', 'IFNA6', 'IFNA5', 'IFNA4', 'IFNL4']"},
             'main_df_link': 's3://serratus-biosamples/mwas_data/Thu_Aug__1_16-11-15_2024/temp_main_df.pickle',
             'job_window': ({'IFNA1': (188, 388), 'IFNA2': (0, 100)}, 300),
             'id': 1,
             'flags': {'IMPLICIT_ZEROS': '0',
                       'GROUP_NONZEROS_ACCEPTANCE_THRESHOLD': '4',
                       'ALREADY_NORMALIZED': '0',
                       'P_VALUE_THRESHOLD': '0.005',
                       'INCLUDE_SKIPPED_GROUP_STATS': '0',
                       'TEST_BLACKLISTED_METADATA_FIELDS': '0',
                       'LOGGING_LEVEL': '2',
                       'USE_LOGGER': '1'}}
    lambda_handler(event, None)
