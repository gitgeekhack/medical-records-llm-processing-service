import io
import boto3

from app.constant import AWS


class S3Utils:
    def __init__(self):
        self.client = boto3.client(service_name='s3', region_name=AWS.BotoClient.AWS_DEFAULT_REGION)

    async def download_object(self, bucket, key, download_path):

        bytes_buffer = io.BytesIO()
        self.client.download_fileobj(Bucket=bucket, Key=key, Fileobj=bytes_buffer)
        file_object = bytes_buffer.getvalue()

        with open(download_path, 'wb') as file:
            file.write(file_object)

    async def upload_object(self, bucket, key, file_object):

        file_object = io.BytesIO(file_object)
        self.client.upload_fileobj(file_object, bucket, key)
        url = f's3://{bucket}/{key}'
        return url

    async def delete_object(self, bucket, key):
        self.client.delete_object(Bucket=bucket, Key=key)

    async def check_s3_path_exists(self, bucket, key):

        response = self.client.list_objects_v2(Bucket=bucket, Prefix=key)
        if 'Contents' not in response or len(response['Contents']) < 1:
            return []
        else:
            return response
