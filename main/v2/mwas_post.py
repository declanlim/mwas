import json
import time
import boto3

def lambda_handler(event: dict, context):
    """subscribes to lambda notifications, waits until it recieves all N of them and then begins the postprocessing"""
    # get the data from event
    link = event['link']  # this is also a unique indicator, so we can filter SNS messages by it
    num_lambdas = event['num_lambdas']

    #
