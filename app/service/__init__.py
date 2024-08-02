import boto3
from botocore.config import Config
from app.constant import AWS

textract_client = boto3.client(
    service_name='textract',
    region_name=AWS.BotoClient.AWS_DEFAULT_REGION,
    config=Config(AWS.BotoClient.TEXTRACT_READ_TIMEOUT))
