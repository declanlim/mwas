import boto3
import json

# Create a Lambda client
lambda_client = boto3.client('lambda')

# Define the event to pass to the Lambda function
event = {
    # Include any necessary event data here
}

# Invoke the Lambda function
response = lambda_client.invoke(
    FunctionName='arn:aws:lambda:us-east-1:797308887321:function:mwas',
    InvocationType='RequestResponse',  # Synchronous invocation
    Payload=json.dumps(event)
)

# Parse the response
response_payload = json.loads(response['Payload'].read())
print(response_payload)
