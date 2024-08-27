"""Lambda function - for statistical tests - for MWAS"""

from mwas_functions import *


def lambda_handler(event: dict, context):
    """Lambda handler for MWAS"""
    start_time = time.perf_counter()
    size = context.memory_limit_in_mb
    # get the data from event
    bioproject_info = event['bioproject_info']
    link = event['link']
    job_window = event['job_window']
    lam_id = event['id']

    CONFIG = Config()

    CONFIG.load_from_json(event['flags'])
    if 'parallel' in event:
        CONFIG.PARALLELIZE = event['parallel']
    CONFIG.set_logger(DIR_SUFFIX + f"log_{bioproject_info['name']}_job{lam_id}.txt")

    CONFIG.log_print(f"Starting MWAS processing for {bioproject_info['name']} job {lam_id}.", 1)
    CONFIG.log_print(f"Lambda info: {context}", 1)

    s3 = boto3.client('s3')
    # get the main_df
    try:
        s3.download_file(S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{MAIN_DF_PICKLE}", DIR_SUFFIX + 'main_df.pickle')
        CONFIG.log_print(f"Downloaded main_df from s3", 1)
        with open(DIR_SUFFIX + 'main_df.pickle', 'rb') as f:
            main_df = pickle.load(f)
        os.remove(DIR_SUFFIX + 'main_df.pickle')

    except Exception as e:
        CONFIG.log_print(f"Error in loading main_df: {e}", 2)
        return sns_publish(500, f"Error in loading main_df: {e}", time.perf_counter() - start_time, size)

    # TODO: for loop here to handle multiple bioprojects in series if the event says so. This will be to optimize for smaller bioprojects, so we don't need to make a new lambda
    #  for each one - which would be especially costly when main_df is large get the bioproject info
    bioproject = load_bioproject_from_dict(CONFIG, bioproject_info)
    # get subset of main_df
    subset_df = main_df[main_df['bio_project'] == bioproject.name]
    del main_df

    # make result file
    CONFIG.OUT_FILE = DIR_SUFFIX + 'result.txt'
    with open(CONFIG.OUT_FILE, 'w') as f:
        f.write("")
    success = bioproject.process_bioproject(subset_df, job_window, lam_id)  # it needs to be the subset of main_df
    if not success:
        return sns_publish(500, f"Error in processing bioproject {bioproject.name} job {lam_id}", time.perf_counter() - start_time, size)
    else:
        CONFIG.log_print(f"Bioproject {bioproject.name} job {lam_id} processed successfully.", 1)

    # upload the result and log files to s3
    try:
        s3.upload_file(DIR_SUFFIX + 'result.txt', S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{OUTPUT_FILES_DIR}/result_{bioproject.name}_job{lam_id}.txt")
        os.remove(DIR_SUFFIX + 'result.txt')

        s3.upload_file(DIR_SUFFIX + f"log_{bioproject_info['name']}_job{lam_id}.txt", S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{PROC_LOG_FILES_DIR}/log_{bioproject.name}_job{lam_id}.txt")

    except Exception as e:
        CONFIG.log_print(f"Error in syncing output: {e}", 2)
        return sns_publish(500, f"Error in putting output to s3: {e}", time.perf_counter() - start_time, size)

    return sns_publish(200, f'MWAS processing completed for {bioproject.name} job {lam_id}', time.perf_counter() - start_time, size)


def sns_publish(status_code, message, time_duration, alias_size):
    """Publishes a message to an SNS topic"""
    sns = boto3.client('sns')
    response = sns.publish(
        TopicArn='arn:aws:sns:us-east-1:797308887321:mwas_sns',
        Message="MWAS Lambda execution details.",  # General description or log message
        Subject='MWAS Lambda Status',
        MessageAttributes={
            'Status_code': {
                'DataType': 'String',
                'StringValue': status_code
            },
            'Message': {
                'DataType': 'String',
                'StringValue': message
            },
            'Time_duration': {
                'DataType': 'Number',
                'StringValue': str(time_duration)  # Convert to string since SNS requires string input
            },
            'Alias_size': {
                'DataType': 'Number',
                'StringValue': str(alias_size)
            }
        }
    )
    return response

#
# if __name__ == '__main__':
#     # indexed_event = {"bioproject_info": {"name": "PRJDB7993",
#     #                              "metadata_file_size": "141241",
#     #                              "n_biosamples": "1990",
#     #                              "n_sets": "1080",
#     #                              "n_permutation_sets": "388",
#     #                              "n_skippable_permutation_sets": "4",
#     #                              "n_groups": "20",
#     #                              "n_skipped_groups": "11",
#     #                              "num_lambda_jobs": "12",
#     #                              "num_conc_procs": "8",
#     #                              "groups": "['IFNA1', 'IFNA2', 'IFNL3', 'IFNL2', 'IFNL1', 'IFNG', 'IFNW1', 'IFNB1', 'IFNA21', 'IFNA17', 'IFNA16', 'IFNA14', 'IFNA13', 'IFNA10', 'IFNA8', 'IFNA7', 'IFNA6', 'IFNA5', 'IFNA4', 'IFNL4']"},
#     #          "link": "Tue_Aug__6_15-20-55_2024",
#     #          "job_window": {"IFNA1": [300, 384], "IFNW1": [0, 216]},
#     #          "id": 1,
#     #          "flags": {"IMPLICIT_ZEROS": "1",
#     #                    "GROUP_NONZEROS_ACCEPTANCE_THRESHOLD": "3",
#     #                    "ALREADY_NORMALIZED": "0",
#     #                    "P_VALUE_THRESHOLD": "0.005",
#     #                    "INCLUDE_SKIPPED_GROUP_STATS": "0",
#     #                    "TEST_BLACKLISTED_METADATA_FIELDS": "0",
#     #                    "LOGGING_LEVEL": "2",
#     #                    "USE_LOGGER": "1",
#     #                    "TIME_LIMIT": "60"},
#     #         "parallel": "1"
#     #         }
#     # lambda_handler(indexed_event, None)
#     #
#     # full_event = {
#     #     "bioproject_info": {
#     #         "name": "PRJNA136121",
#     #         "metadata_file_size": "1723",
#     #         "n_biosamples": "23",
#     #         "n_sets": "10",
#     #         "n_permutation_sets": "3",
#     #         "n_skippable_permutation_sets": "0",
#     #         "n_groups": "20",
#     #         "n_skipped_groups": "0",
#     #         "num_lambda_jobs": "1",
#     #         "num_conc_procs": "6",
#     #         "groups": "everything"
#     #     },
#     #     "link": "Tue_Aug__6_15-20-55_2024",
#     #     "job_window": "full",
#     #     "id": 2,
#     #     "flags": {
#     #         "IMPLICIT_ZEROS": "0",
#     #         "GROUP_NONZEROS_ACCEPTANCE_THRESHOLD": "4",
#     #         "ALREADY_NORMALIZED": "0",
#     #         "P_VALUE_THRESHOLD": "0.005",
#     #         "INCLUDE_SKIPPED_GROUP_STATS": "0",
#     #         "TEST_BLACKLISTED_METADATA_FIELDS": "0",
#     #         "LOGGING_LEVEL": "2",
#     #         "USE_LOGGER": "1"
#     #     }
#     # }
#     # lambda_handler(full_event, None)
#
#     t_event = {'bioproject_info': {'name': 'PRJDB7993',
#                                   'metadata_file_size': '132476',
#                                   'n_biosamples': '1990',
#                                   'n_sets': '1079',
#                                   'n_permutation_sets': '388',
#                                   'n_skippable_permutation_sets': '0',
#                                   'n_groups': '20',
#                                   'n_skipped_groups': '0',
#                                   'num_lambda_jobs': '26',
#                                   'num_conc_procs': '3',
#                                   'groups': 'everything'},
#               'link': 'Tue_Aug__6_15-20-55_2024',
#               'job_window': 'full',
#               'id': 0,
#               'flags': {'IMPLICIT_ZEROS': '0',
#                         'GROUP_NONZEROS_ACCEPTANCE_THRESHOLD': '4',
#                         'ALREADY_NORMALIZED': '0',
#                         'P_VALUE_THRESHOLD': '0.005',
#                         'INCLUDE_SKIPPED_GROUP_STATS': '0',
#                         'TEST_BLACKLISTED_METADATA_FIELDS': '0',
#                         'LOGGING_LEVEL': '2',
#                         'USE_LOGGER': '1',
#                         'TIME_LIMIT': '60'}}
#     lambda_handler(t_event, None)
