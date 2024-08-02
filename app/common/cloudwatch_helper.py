import logging
import os
import sys
import watchtower

from app import logs_client
from app.constant import AWS
from logging_utilities.log_record import LogRecordIgnoreMissing


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        pathname = record.pathname
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.pathname = os.path.relpath(pathname, path).replace('/', '.').replace('\\', '.')
                break
        return True


def get_cloudwatch_handler(log_stream_name):
    return watchtower.CloudWatchLogHandler(
        log_group=AWS.CloudWatch.LOG_GROUP,
        stream_name=log_stream_name,
        boto3_client=logs_client
    )


def get_cloudwatch_logger(project_id=None, document_name=None, log_stream_name=None):
    logging.setLogRecordFactory(LogRecordIgnoreMissing)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    cloudwatch_handler = get_cloudwatch_handler(log_stream_name)
    console_handler = logging.StreamHandler()

    if project_id and document_name:
        formatter = logging.Formatter('[ProjectID: %(project_id)s] - '
                                      '[Document: %(document_name)s] - '
                                      '[Level: %(levelname)s] - '
                                      '[Module: %(pathname)s] - '
                                      '[Function: %(funcName)s] - '
                                      '%(message)s')
    else:
        formatter = logging.Formatter('[Level: %(levelname)s] - '
                                      '[Module: %(pathname)s] - '
                                      '[Function: %(funcName)s] - '
                                      '%(message)s')

    cloudwatch_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    cloudwatch_handler.addFilter(PackagePathFilter())
    logger.addHandler(cloudwatch_handler)
    logger.addHandler(console_handler)

    if project_id and document_name:
        logger = logging.LoggerAdapter(logger, extra={"project_id": project_id, "document_name": document_name})

    return logger
