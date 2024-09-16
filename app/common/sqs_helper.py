import asyncio
import json
import boto3

from app.constant import AWS


class SQSHelper:
    def __init__(self):
        self.aws_region = AWS.BotoClient.AWS_DEFAULT_REGION
        self.sqs = boto3.client('sqs', region_name=self.aws_region)

    async def consume_message(self, queue_url):
        response = self.sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=['SentTimestamp'],
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All'],
            WaitTimeSeconds=20
        )
        try:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            message_body = json.loads(message['Body'])
        except KeyError:
            message_body, receipt_handle = None, None
        return message_body, receipt_handle

    async def publish_message(self, queue_url, message_body):
        response = self.sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
        return response

    async def delete_message(self, queue_url, receipt_handle):
        self.sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
