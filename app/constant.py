import os


class AWS:

    class S3:
        S3_BUCKET = 'ds-medical-insights-extractor'

    class BotoClient:
        AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
        TEXTRACT_READ_TIMEOUT = 3600

    class CloudWatch:
        LOG_GROUP = os.getenv('LOG_GROUP', 'ds-mrs-logs')
        TEXTRACT_RUNNER_STREAM = 'llm-processing-service'
        START_TEXTRACT_STREAM = 'start-textract-async'

    class SQS:
        LLM_OUTPUT_QUEUE = 'https://sqs.ap-south-1.amazonaws.com/851725323009/llm-response-sqs'

    class SNS:
        SNS_TOPIC_ARN = 'arn:aws:sns:ap-south-1:851725323009:textract-async-complete-sns'
        ROLE_ARN = 'arn:aws:iam::851725323009:role/AmazonTextractServiceRole'
