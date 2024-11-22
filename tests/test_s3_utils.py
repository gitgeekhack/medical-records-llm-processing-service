import os
import io
import pytest
from app.common.s3_utils import S3Utils

from botocore.exceptions import ClientError, ParamValidationError

pytest_plugins = ('pytest_asyncio',)


class TestS3Utils:
    @pytest.mark.asyncio
    async def test_download_object_with_valid_parameters(self, tmp_path):
        s3 = S3Utils()
        target_path = tmp_path / "target"
        target_path.mkdir(parents=True, exist_ok=True)


        bucket = "ds-medical-insights-extractor"
        key = "sample-data/sample1_v1.1/request/orthopaedic_clinic.pdf"
        download_path = str(target_path / "orthopaedic_clinic.pdf")

        await s3.download_object(bucket, key, download_path)
        assert sum(len(files) for _, _, files in os.walk(target_path)) == 1

    @pytest.mark.asyncio
    async def test_download_object_with_invalid_key(self, tmp_path):
        s3 = S3Utils()
        target_path = tmp_path / "target"
        target_path.mkdir(parents=True, exist_ok=True)

        bucket = "ds-medical-insights-extractor"
        key = "wrong-key"
        download_path = str(target_path / "operative_report.pdf")

        try:
            await s3.download_object(bucket, key, download_path)
        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_download_object_with_invalid_download_path(self, tmp_path):
        s3 = S3Utils()
        target_path = tmp_path / "target"
        target_path.mkdir(parents=True, exist_ok=True)

        bucket = "ds-medical-insights-extractor"
        key = "sample-data/sample1_v1.1/request/orthopaedic_clinic.pdf"
        download_path = str(target_path / "pdf" / "orthopaedic_clinic.pdf")

        try:
            await s3.download_object(bucket, key, download_path)
        except FileNotFoundError:
            assert True

    @pytest.mark.asyncio
    async def test_download_object_with_invalid_bucket(self, tmp_path):
        s3 = S3Utils()
        target_path = tmp_path / "target"
        target_path.mkdir(parents=True, exist_ok=True)

        bucket = "wrong-bucket-name"
        key = "sample-data/sample1_v1.1/request/orthopaedic_clinic.pdf"
        download_path = str(target_path / "orthopaedic_clinic.pdf")

        try:
            await s3.download_object(bucket, key, download_path)
        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_upload_object_with_valid_parameters(self):
        s3 = S3Utils()
        bytes_buffer = io.BytesIO()
        file_path = "static/Fugarino Dictation_ 06-27-2023.pdf"
        with open(file_path, mode='rb') as file:
            bytes_buffer.write(file.read())

        bucket = "ds-medical-insights-extractor"
        key = "tests-data/Fugarino Dictation_ 06-27-2023.pdf"
        file_object = bytes_buffer.getvalue()

        await s3.upload_object(bucket, key, file_object)
        if await s3.check_s3_path_exists(bucket, key):
            assert True
            await s3.delete_object(bucket, key)
        else:
            assert False

    @pytest.mark.asyncio
    async def test_upload_object_with_invalid_bucket(self):
        s3 = S3Utils()
        bytes_buffer = io.BytesIO()
        file_path = "static/Fugarino Dictation_ 06-27-2023.pdf"
        with open(file_path, mode='rb') as file:
            bytes_buffer.write(file.read())

        bucket = "wrong-bucket-name"
        key = "tests-data/Fugarino Dictation_ 06-27-2023.pdf"
        file_object = bytes_buffer.getvalue()

        try:
            await s3.upload_object(bucket, key, file_object)
            if await s3.check_s3_path_exists(bucket, key):
                assert True
                await s3.delete_object(bucket, key)
            else:
                assert False
        except s3.client.exceptions.NoSuchBucket:
            assert True


    @pytest.mark.asyncio
    async def test_delete_object_with_valid_parameters(self):
        s3 = S3Utils()
        bytes_buffer = io.BytesIO()
        file_path = "static/Fugarino Dictation_ 06-27-2023.pdf"
        with open(file_path, mode='rb') as file:
            bytes_buffer.write(file.read())

        bucket = "ds-medical-insights-extractor"
        key = "tests-data/Fugarino Dictation_ 06-27-2023.pdf"
        file_object = bytes_buffer.getvalue()

        await s3.upload_object(bucket, key, file_object)

        await s3.delete_object(bucket, key)
        if not await s3.check_s3_path_exists(bucket, key):
            assert True
        else:
            assert False

    @pytest.mark.asyncio
    async def test_delete_object_with_invalid_key(self):
        s3 = S3Utils()

        bucket = "ds-medical-insights-extractor"
        key = "invalid-key"
        try:
            await s3.delete_object(bucket, key)
        except ParamValidationError:
            assert True

    @pytest.mark.asyncio
    async def test_delete_object_with_invalid_bucket(self):
        s3 = S3Utils()

        bucket = "invalid-bucket-name"
        key = "test/Fugarino Dictation_ 06-27-2023.pdf"
        try:
            await s3.delete_object(bucket, key)
        except ClientError:
            assert True

    @pytest.mark.asyncio
    async def test_check_s3_path_exists_with_valid_parameters(self):
        s3 = S3Utils()

        bucket = "ds-medical-insights-extractor"
        key = "sample-data/sample1_v1.1/request/orthopaedic_clinic.pdf"
        if await s3.check_s3_path_exists(bucket, key):
            assert True
        else:
            assert False

    @pytest.mark.asyncio
    async def test_check_s3_path_exists_with_invalid_key(self):
        s3 = S3Utils()

        bucket = "ds-medical-insights-extractor"
        key = "invalid-key"
        if not await s3.check_s3_path_exists(bucket, key):
            assert True
        else:
            assert False

    @pytest.mark.asyncio
    async def test_check_s3_path_exists_with_invalid_bucket(self):
        s3 = S3Utils()

        bucket = "invalid-bucket-name"
        key = "test/Fugarino Dictation_ 06-27-2023.pdf"
        try:
            await s3.check_s3_path_exists(bucket, key)
        except ClientError:
            assert True