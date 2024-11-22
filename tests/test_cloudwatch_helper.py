import pytest
from app.common.cloudwatch_helper import get_cloudwatch_logger

pytest_plugins = ('pytest_asyncio',)

class TestCloudwatchLogger:
    @pytest.mark.asyncio
    async def test_cloudwatch_logger_with_params(self):
        project_id = "abc-123-def"
        document_name = "abc.pdf"
        log_stream_name = "your-stream-name"
        logger = get_cloudwatch_logger(project_id, document_name, log_stream_name)
        assert hasattr(logger, 'extra')

    @pytest.mark.asyncio
    async def test_cloudwatch_logger_without_project_id(self):
        logger = get_cloudwatch_logger()
        assert not hasattr(logger, 'extra')