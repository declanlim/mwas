"""Lambda function - for statistical tests - for MWAS"""
import uuid
from datetime import datetime, timedelta
import ast

from mwas_functions import *

CONFIG = Config()
s3 = boto3.client('s3')


def lambda_handler(event: dict, context):
    """Lambda handler for MWAS testing functions"""

    start_time = time.perf_counter()
    size = context.memory_limit_in_mb
    # get the data from event

    print(event)
    print(context)
    if "Records" in event:  # then this is from SNS
        event = json.loads(event['Records'][0]['Sns']['Message'])
        if not isinstance(event['job_window'], str):
            job_window = ast.literal_eval(event['job_window'])
        else:
            job_window = event['job_window']
        bioproject_info = ast.literal_eval(event['bioproject_info'])

    else:
        bioproject_info = event['bioproject_info']
        job_window = event['job_window']

    link = event['link']
    lam_id = event['id']
    expected_jobs = int(event['expected_jobs'])
    try:
        mwas_id = event['mwas_id']
    except KeyError:
        mwas_id = str(uuid.uuid4())

    process_id = f"{bioproject_info['name']}_job{lam_id}"

    try:
        if isinstance(event['flags'], str):
            CONFIG.load_from_json(ast.literal_eval(event['flags']))
        else:
            CONFIG.load_from_json(event['flags'])
        if 'parallel' in event:
            CONFIG.PARALLELIZE = event['parallel']
        CONFIG.set_logger(DIR_SUFFIX + f"log_{bioproject_info['name']}_job{lam_id}.txt")

        CONFIG.log_print(f"recieved message: {event}", 1)
        CONFIG.log_print(f"Starting MWAS processing for {bioproject_info['name']} job {lam_id}.", 1)
        CONFIG.log_print(f"Lambda info: {context}", 1)

        # get the main_df
        try:
            s3.download_file(S3_OUTPUT_DIR_BOTO, f"{link}/{MAIN_DF_PICKLE}", DIR_SUFFIX + 'main_df.pickle')
            CONFIG.log_print(f"Downloaded main_df from s3", 1)
            with open(DIR_SUFFIX + 'main_df.pickle', 'rb') as f:
                main_df = pickle.load(f)
            os.remove(DIR_SUFFIX + 'main_df.pickle')

        except Exception as e:
            CONFIG.log_print(f"Error in loading main_df: {e}", 2)
            return exit_handler(500, f"Error in loading main_df: {e}", time.perf_counter() - start_time, size, process_id, mwas_id, expected_jobs, link)

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
            return exit_handler(500, f"Error in processing bioproject {bioproject.name} job {lam_id}", time.perf_counter() - start_time, size, process_id, mwas_id, expected_jobs, link)
        else:
            CONFIG.log_print(f"Bioproject {bioproject.name} job {lam_id} processed successfully.", 1)

        return exit_handler(200, f'MWAS processing completed for {bioproject.name} job {lam_id}', time.perf_counter() - start_time, size, process_id, mwas_id, expected_jobs, link)
    except Exception as e:
        CONFIG.log_print(f"Error in processing: {e}", 2)
        return exit_handler(500, f"Error in processing: {e}", time.perf_counter() - start_time, size, process_id, mwas_id, expected_jobs, link)


def exit_handler(status_code, message, time_duration, alias_size, process_id, mwas_id, expected_jobs, link):
    """Exit function for testing"""
    # upload the result and log files to s3
    try:
        s3.upload_file(DIR_SUFFIX + 'result.txt', S3_OUTPUT_DIR_BOTO, f"{link}/{OUTPUT_FILES_DIR}/result_{process_id}.txt")
        CONFIG.log_print(f"Uploaded output file to s3", 1)
        # os.remove(DIR_SUFFIX + 'result.txt')
    except Exception as e:
        CONFIG.log_print(f"Error in uploading output file: {e}", 2)
    try:
        s3.upload_file(DIR_SUFFIX + f"log_{process_id}.txt", S3_OUTPUT_DIR_BOTO, f"{link}/{PROC_LOG_FILES_DIR}/log_{process_id}.txt")
        CONFIG.log_print(f"Uploaded log file to s3", 1)
    except Exception as e:
        CONFIG.log_print(f"Error in uploading log file: {e}", 2)
    return dynamoDB_store(status_code, message, time_duration, alias_size, process_id, mwas_id, expected_jobs, link)


def dynamoDB_store(status_code, message, time_duration, alias_size, process_id, mwas_id, expected_jobs, link):
    """Publishes a message to an SNS topic"""
    CONFIG.log_print(f"Storing exit message in dynamoDB", 1)
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('mwas_notification_handler')
    item = {
        'mwas_id': str(mwas_id),  # unique identifier for the mwas run given by the preprocessing function (so all lambdas have the same mwas_id), this is the dynamoDB partition key
        'lambda_id': str(process_id),  # unique identifier for the lambda, this is the dynamoDB sort key
        'status_code': status_code,  # success or failure
        'message': message,  # exit message
        'time_duration': str(time_duration),  # informs how long the lambda took to run (seconds), to help compute cost
        'alias_size': str(alias_size),  # informs what lambda size was used, to help compute cost
        'TTL': int((datetime.now() + timedelta(hours=1)).timestamp())   # expiration date - when dynamoDB will auto remove item
    }
    table.put_item(Item=item)
    CONFIG.log_print(f"Stored in dynamoDB: {item}", 1)

    if CONFIG.IGNORE_DYNAMO:
        CONFIG.log_print(f"Flag set to Ignoring dynamoDB, so we don't check the dynamoDB message count.", 1)
        return item
    # scan the table to see if this was the last lambda to finish
    dynamodb_client = boto3.client('dynamodb')
    response = dynamodb_client.query(
        TableName='mwas_notification_handler',
        KeyConditionExpression=f"mwas_id = :pk_value",
        ExpressionAttributeValues={":pk_value": {"S": mwas_id}},
        Select="COUNT"
    )
    CONFIG.log_print(f"Count of items in dynamoDB: {response['Count']}", 1)
    if response['Count'] >= expected_jobs:
        # all lambdas have finished
        # send a message to the SNS topic to notify the user
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:797308887321:mwas_sns',
            Message=json.dumps({'mwas_id': mwas_id, 'link': link, 'expected_jobs': expected_jobs,
                                'status': 'All lambdas have finished'}),  # Message is important to be written like this, or else post proc won't get triggered
            Subject=f"MWAS run {mwas_id} has completed",
            MessageAttributes={
                'status': {
                    'DataType': 'String',
                    'StringValue': 'All lambdas have finished'
                },
                'mwas_id': {
                    'DataType': 'String',
                    'StringValue': mwas_id
                },
                'link': {
                    'DataType': 'String',
                    'StringValue': link
                },
                'expected_jobs': {
                    'DataType': 'Number',
                    'StringValue': str(expected_jobs)
                }
            }
        )
        CONFIG.log_print(f"This was the last lambda to finish. Notified the SNS topic to trigger postprocessing", 1)
    else:
        CONFIG.log_print(f"Still {expected_jobs - response['Count']} lambdas to finish", 1)

    return item


def test_storing_in_dynamoDB():
    """Tests storing in dynamoDB"""
    dynamoDB_store(200, "Test message", 0.32, 1024, str(uuid.uuid4()), 'test', 2, 'test_link')
