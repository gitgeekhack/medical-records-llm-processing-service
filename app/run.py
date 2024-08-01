import json
import os
import traceback
import asyncio
from app.service.llm_proessing import LLMProcessing
from app.common.cloudwatch_helper import setup_cloudwatch_logging

logger = setup_cloudwatch_logging()
input_message = os.getenv('INPUT_MESSAGE')
if not input_message:
    logger.info('Configuration incomplete. Please configure INPUT_MESSAGE environment variable.')
    exit(0)


async def get_project_id_and_document(document_path):
    document_name = os.path.basename(document_path)
    project_id = document_path.split('/')[2]
    return project_id, document_name


async def main():
    global logger
    try:
        input_message_dict = json.loads(input_message)
        project_id, document_name = await get_project_id_and_document(input_message_dict['DocumentLocation']['S3ObjectName'])
        logger = setup_cloudwatch_logging(project_id, document_name)
        # logger.info(f"Input message: {input_message}")
        llm_processor = LLMProcessing(logger, project_id, document_name)
        await llm_processor.process_doc(input_message_dict)
    except Exception as e:
        logger.error('%s -> %s' % (e, traceback.format_exc()))


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
