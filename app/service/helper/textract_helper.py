import json
import os
import tempfile
import time

import boto3

from app.common.s3_utils import S3Utils
from app.constant import AWS, MedicalInsights

textract_client = boto3.client('textract', region_name=AWS.BotoClient.AWS_DEFAULT_REGION)


class TextractHelper:
    def __init__(self, logger):
        self.logger = logger
        self.s3_utils = S3Utils()

    async def get_text(self, job_id, page_wise_text={}, next_token=None):
        if next_token:
            textract_response = textract_client.get_document_text_detection(JobId=job_id, NextToken=next_token)
        else:
            textract_response = textract_client.get_document_text_detection(JobId=job_id)

        for block in textract_response['Blocks']:
            if block['BlockType'] == 'LINE':
                page_wise_text['page_' + str(block['Page'])] = (
                        page_wise_text.get('page_' + str(block['Page']), '') + block['Text'] + ' ')

        if 'NextToken' in textract_response:
            await self.get_text(job_id, page_wise_text, textract_response['NextToken'])

        return page_wise_text

    async def get_page_wise_text(self, input_message):
        start_time = time.time()
        self.logger.info("Text Extraction from document is started...")

        file_path = input_message['DocumentLocation']['S3ObjectName']
        pdf_name = os.path.basename(file_path)
        temp_dir = tempfile.TemporaryDirectory()
        local_textract_path = os.path.join(os.path.join(temp_dir.name, f'{pdf_name}_text.json'))
        s3_textract_path = os.path.join(os.path.dirname(os.path.dirname(file_path)),
                                        MedicalInsights.TEXTRACT_FOLDER_NAME, f'{pdf_name}_text.json')
        response = await self.s3_utils.check_s3_path_exists(bucket=AWS.S3.S3_BUCKET, key=s3_textract_path)

        if response:
            self.logger.info("Reading textract response from the cache...")
            await self.s3_utils.download_object(AWS.S3.S3_BUCKET, s3_textract_path, local_textract_path)
            with open(local_textract_path, 'r') as file:
                json_data = json.loads(file.read())
            page_wise_text = json_data
        else:
            self.logger.info("Using GetDocumentTextDetection api...")
            page_wise_text = await self.get_text(input_message['JobId'])
            result = json.dumps(page_wise_text)
            result = result.encode("utf-8")
            await self.s3_utils.upload_object(AWS.S3.S3_BUCKET, s3_textract_path, result)

        self.logger.info(f"Get text using textract completed in {time.time() - start_time} seconds.")

        return page_wise_text
