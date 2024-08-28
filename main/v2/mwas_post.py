import time
from datetime import datetime, timedelta
import boto3

S3_BUCKET_BOTO = 'serratus-biosamples'
S3_OUTPUT_DIR_BOTO = 'mwas_data'
PROGRESS_FILE = 'progress.json'
OUTPUT_CSV_FILE = 'mwas_output.csv'
OUTPUT_FILES_DIR = 'outputs'

LAMBDA_PRICING = lambda x: (x / 1024) * 0.00001667  # subject to change, but this is 0.00001667 / GB-second.

dynamodb = boto3.client('dynamodb')


def lambda_handler(event: dict, context):
    """subscribes to lambda notifications, waits until it recieves all N of them and then begins the postprocessing"""
    # get the data from event
    mwas_id = event['mwas_id']
    wait_time = event['wait_time']
    check_interval = event['check_interval']
    expected_jobs = event['expected_jobs']
    link = event['link']

    # listening for messages from the dynamoDB
    start_time = datetime.now()
    got_all_jobs = False
    messages = []
    while datetime.now() - start_time < timedelta(seconds=wait_time):
        messages = fetch_messages(mwas_id)
        print(messages)
        if len(messages) >= expected_jobs:
            got_all_jobs = True
            break
        time.sleep(check_interval)
    successes, fails, total_cost = process_messages(messages)
    print("processed this many messages: ", len(messages), " successes: ", successes, " fails: ", fails, " total cost: ", total_cost, "got all jobs: ", got_all_jobs)

    # start the postprocessing
    postprocessing(link)
    return {
        'statusCode': 200,
        'message': f"processed this many messages: {len(messages)}, successes: {successes}, fails: {fails}, total cost: {total_cost}, got all jobs: {got_all_jobs}"
    }


def process_messages(messages):
    """see mwas_lambda.py for json structure of messages"""
    successes, fails, total_cost = 0, 0, 0
    for message in messages:
        if message['status_code']['N'] == 200:
            successes += 1
        else:
            fails += 1
        alias_size, time_duration = int(message['alias_size']['S']), float(message['time_duration']['S'])
        total_cost += time_duration * LAMBDA_PRICING(alias_size)
        print("recieved message: ", message)

        delete_message(message['lambda_id']['S'], message['mwas_id']['S'])
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
    """delete the message from the dynamoDB"""
    dynamodb.delete_item(
        TableName='mwas_notification_handler',
        Key={
            'lambda_id': {'S': lambda_id},
            'mwas_id': {'S': mwas_id}
        }
    )


def postprocessing(link):
    """postprocessing of the results"""
    print("postprocessing of the results")
    # get results from s3. This means going through all files stored in s3.upload_file(DIR_SUFFIX + 'result.txt', S3_BUCKET_BOTO, f"{S3_OUTPUT_DIR_BOTO}/{link}/{OUTPUT_FILES_DIR}/result_{bioproject.name}_job{lam_id}.txt")
    s3 = boto3.client('s3')
    s3_dir = f"{S3_OUTPUT_DIR_BOTO}/{link}/{OUTPUT_FILES_DIR}/"

    # download every object in this dir by syncing that dir with a local dir
    # create a local dir in tmp
    os.

    s3.download_file(S3_BUCKET_BOTO, s3_dir, local_dir)
