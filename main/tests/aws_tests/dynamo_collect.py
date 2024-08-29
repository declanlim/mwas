import boto3
import time
from datetime import datetime, timedelta

dynamodb = boto3.client('dynamodb')


def fetch_messages(mwas_id_value):
    # Use a filter expression to only fetch items with the specific mwas_id
    # response = dynamodb.scan(
    #     TableName='mwas_notification_handler',
    #     FilterExpression='mwas_id = :mwas_id_val',
    #     ExpressionAttributeValues={':mwas_id_val': {'S': mwas_id_value}}
    # )
    # return response['Items']

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
    dynamodb.delete_item(
        TableName='mwas_notification_handler',
        Key={
            'lambda_id': {'S': lambda_id},
            'mwas_id': {'S': mwas_id}
        }
    )


def process_messages(messages):
    for message in messages:
        # Example processing logic
        print(f"Processing message {message['lambda_id']}")
        # Additional processing steps...
        print(f"message: {message}")
        delete_message(message['lambda_id']['S'], message['mwas_id']['S'])


if __name__ == '__main__':
    # Monitoring loop
    start_time = datetime.now()
    threshold = 2  # Example threshold
    time_limit = 10
    mwas_id_value = 'test'  # Set the specific mwas_id value you want to filter by

    while True:
        messages = fetch_messages(mwas_id_value)
        print(messages)
        if len(messages) >= threshold or datetime.now() - start_time > timedelta(seconds=time_limit):
            break
        time.sleep(1)  # Check every second
    process_messages(messages)
    print("processed this many messages: ", len(messages))
