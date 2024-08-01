import boto3
from app.constant import AWS

logs_client = boto3.client('logs', region_name=AWS.BotoClient.AWS_DEFAULT_REGION)
