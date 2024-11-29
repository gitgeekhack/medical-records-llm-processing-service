import json
import pytest
from app.common.sqs_helper import SQSHelper

from botocore.exceptions import ClientError, ParamValidationError

pytest_plugins = ('pytest_asyncio',)


class TestSQSHelper:
    @pytest.mark.asyncio
    async def test_consume_message_with_valid_parameters(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"

        message_body = {"test_msg": "Test Message for the SQS"}
        message_body = json.dumps(message_body)

        await sqs.publish_message(queue_name, message_body)
        message_body, receipt_handle = await sqs.consume_message(queue_name)

        if message_body:
            await sqs.delete_message(queue_name, receipt_handle)
            assert True

        else:
            assert False

    @pytest.mark.asyncio
    async def test_consume_message_with_empty_message_body(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"

        message_body, receipt_handle = await sqs.consume_message(queue_name)

        if message_body:
            assert False
        else:
            assert True

    @pytest.mark.asyncio
    async def test_consume_message_with_invalid_queue_url(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async"

        try:
            await sqs.consume_message(queue_name)
            assert False

        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_publish_message_with_valid_parameters(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"

        message_body = {"test_msg": "Test Message for the SQS"}
        message_body = json.dumps(message_body)

        await sqs.publish_message(queue_name, message_body)
        message_body, receipt_handle = await sqs.consume_message(queue_name)

        if message_body:
            await sqs.delete_message(queue_name, receipt_handle)
            assert True

        else:
            assert False

    @pytest.mark.asyncio
    async def test_publish_message_with_invalid_queue_url(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async"
        message_body = {"test_msg": "Test Message for the SQS"}
        message_body = json.dumps(message_body)

        try:
            await sqs.publish_message(queue_name, message_body)
            assert False

        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_publish_message_with_invalid_message_body(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"
        message_body = {"test_msg": "Test Message for the SQS"}

        try:
            await sqs.publish_message(queue_name, message_body)
            assert False

        except ParamValidationError:
            assert True

    @pytest.mark.asyncio
    async def test_delete_message_with_valid_parameters(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"

        message_body = {"test_msg": "Test Message for the SQS"}
        message_body = json.dumps(message_body)

        await sqs.publish_message(queue_name, message_body)
        message_body, receipt_handle = await sqs.consume_message(queue_name)

        if message_body:
            await sqs.delete_message(queue_name, receipt_handle)
            assert True

        else:
            assert False

    @pytest.mark.asyncio
    async def test_delete_message_with_invalid_receipt_handle(self, tmp_path):
        sqs = SQSHelper()

        queue_name = "completed-textract-async-sqs"
        receipt_handle = "invalid receipt handle"

        try:
            await sqs.delete_message(queue_name, receipt_handle)
            assert False

        except ClientError:
            assert True