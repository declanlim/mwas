import json
import boto3


def lambda_handler(event, context):
    """Lambda to help mwas_main to dispatch jobs faster. multiple of these will run in parallel"""
    print(event)
    print(context)

    # get the data from event
    # note event is a job_slice and mwas_id
    mwas_id = event['mwas_id']
    job_slice = event['jobs']
    expected_jobs = event['expected_jobs']
    link = event['link']
    flags = event['flags']

    LAMBDA_CLIENT = boto3.client('lambda')
    for job in job_slice:
        # invoke the lambda
        func_choice = job['func']
        job['expected_jobs'] = expected_jobs
        job['mwas_id'] = mwas_id
        job['link'] = link
        job['flags'] = flags

        print(f"invoking a {func_choice}")
        print("job: ", job)
        response = LAMBDA_CLIENT.invoke(
            FunctionName=f'arn:aws:lambda:us-east-1:797308887321:function:{func_choice}',
            InvocationType='Event',
            Payload=json.dumps(job)
        )
