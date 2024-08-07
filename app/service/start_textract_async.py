import json
import traceback
from botocore.exceptions import ClientError

from app.service import textract_client
from app.common.sqs_helper import SQSHelper
from app.constant import AWS


class TextractProcessor:
    def __init__(self, logger, project_id, document_name):
        self.logger = logger
        self.project_id = project_id
        self.document_name = document_name
        self.sqs_helper = SQSHelper()
        self.textract_client = textract_client

    def start_document_text_detection(self, bucket, document):
        try:
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket,
                        'Name': document
                    }
                },
                NotificationChannel={
                    'SNSTopicArn': AWS.SNS.SNS_TOPIC_ARN,
                    'RoleArn': AWS.SNS.ROLE_ARN
                }
            )
            return response
        except ClientError:
            self.logger.error(traceback.format_exc())
            raise

    async def process_document(self, document_path):
        try:
            self.logger.info(f"Processing document: {self.document_name}")
            job_id = self.start_document_text_detection(AWS.S3.S3_BUCKET, document_path)
            self.logger.info(f"Successfully started Textract job with ID: {job_id['JobId']}")
            return job_id

        except:
            # Prepare error message for SQS
            error_message = {
                "status": "FAILED",
                "project_id": self.project_id,
                "document_name": self.document_name,
                "output_path": None
            }

            await self.sqs_helper.publish_message(AWS.SQS.LLM_OUTPUT_QUEUE, json.dumps(error_message))
            self.logger.info(f"Published error message to the queue: {AWS.SQS.LLM_OUTPUT_QUEUE.split('/')[-1]}")