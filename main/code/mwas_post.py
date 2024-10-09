import json
import os
import time
# from datetime import datetime, timedelta
import boto3
import subprocess

S3_BUCKET_BOTO = 'serratus-biosamples'
S3_OUTPUT_DIR_BOTO = 'mwas-user-dump'
PROGRESS_FILE = 'progress.json'
OUTPUT_CSV_FILE = 'mwas_output.csv'
OUTPUT_FILES_DIR = 'outputs'

dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')


def lambda_handler(event: dict, context):
    """recieves an event notification from the last lambda (mwas_lambda.py) and processes the results, i.e. combines the result files"""
    post_start_time = time.time()

    # get the data from event (sns message)
    try:
        sns_message = event['Records'][0]['Sns']['Message']
        message = json.loads(sns_message)
    except KeyError:
        message = event

    mwas_id = message['mwas_id']
    link = message['link']
    expected_jobs = int(message['expected_jobs'])
    print("recieved message:", message)

    messages = fetch_messages(mwas_id)
    successes, fails, total_cost = process_messages(messages)
    print("processed this many messages: ", len(messages), " successes: ", successes, " fails: ", fails, " total cost: ", total_cost)

    # get progress.json from s3
    s3.download_file(S3_OUTPUT_DIR_BOTO, f"{link}/{PROGRESS_FILE}", f"/tmp/{PROGRESS_FILE}")
    print("downloaded progress.json")
    with open(f"/tmp/{PROGRESS_FILE}", 'r') as f:
        progress = json.load(f)
        if progress['status'] == "post-processing" or progress['status'] == "MWAS COMPLETED":
            print("postprocessing already done. Exiting.")
            return {
                'statusCode': 200,
                'message': f"postprocessing already done, likely due to redundant triggering via alarm for safety invoke."
            }
    with open(f"/tmp/{PROGRESS_FILE}", 'w') as f:
        start_time = time.time()
        json.dump({
            'status': "post-processing",
            'num_bioprojects': progress['num_bioprojects'],
            'num_lambda_jobs': progress['num_lambda_jobs'],
            'num_permutation_tests': progress['num_permutation_tests'],
            'num_jobs_completed': str(len(messages)),
            'time_elapsed': str(time.time() - start_time),
            'start_time': str(start_time),
            'successes': str(successes),
            'fails': str(fails),
            'total_cost': str(total_cost)
        }, f)
    # sync progress.json to s3
    s3.upload_file(f"/tmp/{PROGRESS_FILE}", S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{PROGRESS_FILE}")
    print("updated progress.json and now starting postprocessing combining")

    # start the postprocessing
    permutation_tests, num_bioprojects = postprocessing(link)
    with open(f"/tmp/{PROGRESS_FILE}", 'w') as f:
        json.dump({
            'status': "MWAS COMPLETED",
            'num_bioprojects': str(num_bioprojects),
            'num_lambda_jobs': progress['num_lambda_jobs'],
            'num_permutation_tests': str(permutation_tests),
            'num_jobs_completed': str(len(messages)),
            'time_elapsed': str(time.time() - start_time),
            'start_time': str(start_time),
            'successes': str(successes),
            'fails': str(fails),
            'total_cost': str(total_cost)
        }, f)
        # sync progress.json to s3
    s3.upload_file(f"/tmp/{PROGRESS_FILE}", S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{PROGRESS_FILE}")

    print("postprocessing done. Time spent on postprocessing: ", time.time() - post_start_time)
    print("MWAS runtime: ", time.time() - start_time)
    return {
        'statusCode': 200,
        'message': f"processed this many messages: {len(messages)}, successes: {successes}, fails: {fails}, total cost: {total_cost}"
    }


def lamdba_pricing(x: int):
    """subject to change, but this is 0.00001667 / GB-second."""
    return (x / 1024) * 0.00001667


def process_messages(messages):
    """see mwas_lambda.py for json structure of messages"""
    successes, fails, total_cost = 0, 0, 0
    for message in messages:
        if message['status_code']['N'] == '200':
            successes += 1
        else:
            fails += 1
        alias_size, time_duration = int(message['alias_size']['S']), float(message['time_duration']['S'])
        total_cost += time_duration * lamdba_pricing(alias_size)
        print("recieved message: ", message)

        # delete_message(message['lambda_id']['S'], message['mwas_id']['S'])
    return successes, fails, total_cost


def fetch_messages(mwas_id_value):
    """scan and fetch all messages corresponding to this mwas run, as per mwas_id, from the dynamoDB"""
    items = []
    last_evaluated_key = None

    while True:
        if last_evaluated_key:
            response = dynamodb.scan(
                TableName='mwas_notification_handler',
                FilterExpression='mwas_id = :mwas_id_val',
                ExpressionAttributeValues={':mwas_id_val': {'S': mwas_id_value}},
                ExclusiveStartKey=last_evaluated_key
            )
        else:
            response = dynamodb.scan(
                TableName='mwas_notification_handler',
                FilterExpression='mwas_id = :mwas_id_val',
                ExpressionAttributeValues={':mwas_id_val': {'S': mwas_id_value}}
            )
        items.extend(response['Items'])

        # Check if there's more data to retrieve
        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break

    return items


def delete_message(lambda_id, mwas_id):
    """delete the message from the dynamoDB
    (not used)"""
    dynamodb.delete_item(
        TableName='mwas_notification_handler',
        Key={
            'lambda_id': {'S': lambda_id},
            'mwas_id': {'S': mwas_id}
        }
    )


def postprocessing(link):
    """postprocessing of the results"""
    print(f"combining the result files from all the lambdas for this mwas run {link}")
    s3_dir = f"{link}/{OUTPUT_FILES_DIR}/"
    files_dir = '/tmp/mwas_results'
    os.mkdir(files_dir)
    try:
        print(s3_dir)
        response = s3.list_objects_v2(Bucket=S3_OUTPUT_DIR_BOTO, Prefix=s3_dir)
        print("num of objects in s3: ", len(response.get('Contents', [])))
        print("first 2 objects: ", response.get('Contents', [])[:2])
        for obj in response.get('Contents', []):
            file_name = obj['Key'].split('/')[-1]
            if file_name == '':  # this is the directory itself... a weird thing that s3 does
                continue
            print(f"downloading {file_name}, i.e. {obj['Key']}")
            s3.download_file(S3_OUTPUT_DIR_BOTO, obj['Key'], f"{files_dir}/{file_name}")
            print(f"downloaded {file_name}")
    except Exception as e:
        print(f"Error in syncing output: {e}")
        return
    OUT_COLS = [
        'bioproject', 'group', 'metadata_field', 'metadata_value', 'test', 'num_true', 'num_false', 'mean_rpm_true', 'mean_rpm_false',
        'sd_rpm_true', 'sd_rpm_false', 'fold_change', 'test_statistic', 'p_value', 'true_biosamples', 'false_biosamples', 'test_time_duration', 'job_id'
    ]
    num_permutation_tests = 0
    bioprojects = set()
    with open(f"/tmp/{OUTPUT_CSV_FILE}", 'w') as final_output:
        final_output.write(','.join(OUT_COLS) + '\n')
        for file in os.listdir(files_dir):
            if file.endswith('.txt'):
                with open(f"{files_dir}/{file}", 'r') as result_file:
                    lines = result_file.readlines()
                    for line in lines:
                        comma_index = line.find(',')
                        if comma_index != -1:
                            final_output.write(line)
                            if 'permutation' in line:
                                num_permutation_tests += 1
                                bioproject_name = line[:comma_index]
                                bioprojects.add(bioproject_name)
    s3.upload_file(f"/tmp/{OUTPUT_CSV_FILE}", S3_OUTPUT_DIR_BOTO, f"{link}/{OUTPUT_CSV_FILE}")
    return num_permutation_tests, len(bioprojects)
