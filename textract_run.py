import json
import os
import traceback
import asyncio

from app.service.start_textract_async import TextractProcessor
from app.common.cloudwatch_helper import get_cloudwatch_logger
from app.constant import AWS
from app.common.utils import get_project_id_and_document

logger = get_cloudwatch_logger(log_stream_name=AWS.CloudWatch.START_TEXTRACT_STREAM)
input_message = os.getenv('INPUT_MESSAGE')
if not input_message:
    logger.info('Configuration incomplete. Please configure INPUT_MESSAGE environment variable.')
    exit(0)

if not AWS.SQS.LLM_OUTPUT_QUEUE_URL:
    logger.info('Configuration incomplete. Please configure LLM_OUTPUT_QUEUE_URL environment variable.')
    exit(0)

if not AWS.SNS.SNS_TOPIC_ARN:
    logger.info('Configuration incomplete. Please configure SNS_TOPIC_ARN environment variable.')
    exit(0)

if not AWS.SNS.ROLE_ARN:
    logger.info('Configuration incomplete. Please configure ROLE_ARN environment variable.')
    exit(0)

async def main():
    global logger
    try:
        input_message_dict = json.loads(input_message)
        document_path = input_message_dict.get('document_path', '')
        if not document_path:
            logger.error("document_path not found in INPUT_MESSAGE")
            return

        if not document_path.lower().endswith('.pdf'):
            logger.error("The document path does not appear to be a PDF file")
            return

        project_id, document_name = await get_project_id_and_document(document_path=document_path)
        logger = get_cloudwatch_logger(project_id, document_name, log_stream_name=AWS.CloudWatch.START_TEXTRACT_STREAM)
        textract_processor = TextractProcessor(logger, project_id, document_name)
        await textract_processor.process_document(document_path=document_path)
    except Exception as e:
        logger.error('%s -> %s' % (e, traceback.format_exc()))


if __name__ == "__main__":
    asyncio.run(main())
