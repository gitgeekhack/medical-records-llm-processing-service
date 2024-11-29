import logging

import pytest
from botocore.exceptions import ClientError

from app.common.cloudwatch_helper import get_cloudwatch_logger
from app.service.start_textract_async import TextractProcessor

pytest_plugins = ('pytest_asyncio',)


class TestStartTextractAsync:
    @pytest.mark.asyncio
    async def test_start_document_text_detection_with_valid_parameters(self):

        logger = get_cloudwatch_logger()
        project_id = "textract-test"
        document_name = "Fenn Dictation_ 09-06-2022.pdf"
        textract_processor = TextractProcessor(logger, project_id, document_name)

        bucket = "ds-medical-insights-extractor"
        key = "test-data/Fenn Dictation_ 09-06-2022.pdf"

        # update the env variables of AWS SNS and LLM_OUTPUT_QUEUE_URL in pytest.ini
        response = textract_processor.start_document_text_detection(bucket, key)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    @pytest.mark.asyncio
    async def test_start_document_text_detection_with_invalid_download_path(self, tmp_path):
        # Use a temporary logger for this test
        temp_logger = logging.getLogger("test_logger")
        temp_logger.setLevel(logging.CRITICAL)

        project_id = "textract-test"
        document_name = "Fenn Dictation_ 09-06-2022.pdf"
        textract_processor = TextractProcessor(temp_logger, project_id, document_name)

        bucket = "ds-medical-insights-extractor"
        key = "invalid-download-path"

        try:

            # update the env variables of AWS SNS and LLM_OUTPUT_QUEUE_URL in pytest.ini
            textract_processor.start_document_text_detection(bucket, key)
            assert False
        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_start_document_text_detection_with_invalid_bucket(self, tmp_path):
        # Use a temporary logger for this test
        temp_logger = logging.getLogger("test_logger")
        temp_logger.setLevel(logging.CRITICAL)

        project_id = "textract-test"
        document_name = "Fenn Dictation_ 09-06-2022.pdf"
        textract_processor = TextractProcessor(temp_logger, project_id, document_name)

        bucket = "invalid-bucket"
        key = "test-data/Fenn Dictation_ 09-06-2022.pdf"

        try:
            # update the env variables of AWS SNS and LLM_OUTPUT_QUEUE_URL in pytest.ini
            textract_processor.start_document_text_detection(bucket, key)
            assert False
        except ClientError:
            assert True
