import asyncio
import json
import os
import traceback

from app.common.cloudwatch_helper import get_cloudwatch_logger
from app.common.utils import get_project_id_and_document
from app.constant import AWS
from app.service.llm_processing import LLMProcessing

logger = get_cloudwatch_logger(log_stream_name=AWS.CloudWatch.LLM_PROCESSING_STREAM)
input_message = os.getenv('INPUT_MESSAGE')
if not input_message:
    logger.info('Configuration incomplete. Please configure INPUT_MESSAGE environment variable.')
    exit(0)


async def main():
    global logger
    try:
        input_message_dict = json.loads(input_message)
        project_id, document_name = await get_project_id_and_document(
            input_message_dict['DocumentLocation']['S3ObjectName'])
        logger = get_cloudwatch_logger(project_id, document_name, AWS.CloudWatch.LLM_PROCESSING_STREAM)
        logger.info(f"Input message: {input_message}")
        llm_processor = LLMProcessing(logger, project_id, document_name)
        await llm_processor.process_doc(input_message_dict)
    except Exception as e:
        logger.error('%s -> %s' % (e, traceback.format_exc()))


if __name__ == "__main__":
    asyncio.run(main())
