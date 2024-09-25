import asyncio
import os
import traceback
import json

from app.common.cloudwatch_helper import get_cloudwatch_logger
from app.constant import AWS
from app.service.llm_processing import LLMProcessing

logger = get_cloudwatch_logger(log_stream_name=AWS.CloudWatch.LLM_PROCESSING_STREAM)
input_message = os.getenv('INPUT_MESSAGE')
if not input_message:
    logger.info('Configuration incomplete. Please configure INPUT_MESSAGE environment variable.')
    exit(0)

if not AWS.SQS.LLM_OUTPUT_QUEUE_URL:
    logger.info('Configuration incomplete. Please configure LLM_OUTPUT_QUEUE_URL environment variable.')
    exit(0)


async def main():
    global logger
    try:
        input_message_dict = json.loads(input_message)
        document_name = os.path.basename(input_message_dict['document_name'])
        logger = get_cloudwatch_logger(document_name, AWS.CloudWatch.LLM_PROCESSING_STREAM)
        logger.info(f"Input message: {input_message}")
        llm_processor = LLMProcessing(logger, document_name)
        await llm_processor.process_doc(input_message_dict)
    except Exception as e:
        logger.error('%s -> %s' % (e, traceback.format_exc()))


if __name__ == "__main__":
    asyncio.run(main())
