import json
import boto3

s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """generate a presigned url for the user to successfully upload the input file to S3 via curl"""
    try:
        if 'body' not in event:
            raise ValueError("Event does not contain 'body'")

        body = event['body']
        if isinstance(body, str):  # addresses a string key issue
            body = json.loads(body)
        if 'hash' not in body or 'bucket_name' not in body:
            raise ValueError("Body does not contain 'hash' or 'bucket_name'")

        hash_value = body['hash']
        bucket_name = body['bucket_name']
        # check if this hashed folder already exists in s3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=hash_value)
            exit_handler({
                'statusCode': 500,
                'message': 'File already exists in S3'
            })
        except Exception as e:
            print("File does not exist yet. This is good. ", e)

        presigned_url = s3_client.generate_presigned_url(
            'put_object', Params={'Bucket': bucket_name, 'Key': hash_value}, ExpiresIn=300)
        exit_handler({
            'statusCode': 200,
            'presigned_url': presigned_url,
            'hash': hash_value
        })

    except Exception as e:
        exit_handler({
            'statusCode': 500,
            'message': f'Error in getting hash and bucket_name from event. {e}'
        })


def exit_handler(item):
    """Exit function"""
    print(item)
    body = json.dumps(item)
    return {
        'statusCode': item['statusCode'],
        'body': body,
        'headers': {
            'Content-Type': 'application/json'
        }
    }
