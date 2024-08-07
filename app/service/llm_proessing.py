import asyncio
import json
import logging
import os
import time
import traceback
from concurrent import futures

from app.business_rule_exception import TextExtractionFailed
from app.common.s3_utils import S3Utils
from app.common.sqs_helper import SQSHelper
from app.constant import AWS, MedicalInsights
from app.service.helper.textract_helper import TextractHelper
from app.service.nlp_extractor.document_summarizer import DocumentSummarizer
from app.service.nlp_extractor.encounters_extractor import EncountersExtractor
from app.service.nlp_extractor.entity_extractor import get_extracted_entities
from app.service.nlp_extractor.phi_and_doc_type_extractor import PHIAndDocTypeExtractor

logging.getLogger("faiss").setLevel(logging.WARNING)


class LLMProcessing:
    def __init__(self, logger, project_id, document_name):
        self.logger = logger
        self.project_id = project_id
        self.document_name = document_name
        self.sqs_helper = SQSHelper()
        self.s3_utils = S3Utils()
        self.textract_helper = TextractHelper(logger)

    async def get_summary(self, data):
        """ This method is used to get document summary """

        start_time = time.time()
        self.logger.info("Summary generation is started...")
        document_summarizer = DocumentSummarizer(self.logger)
        summary = await document_summarizer.get_summary(data)
        self.logger.info(f"Summary is generated in {time.time() - start_time} seconds.")
        return summary

    def get_summary_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_summary(data))
        return x

    async def get_entities(self, data):
        """ This method is used to get entities from document """

        start_time = time.time()
        self.logger.info("Entity Extraction is started...")
        entities = await get_extracted_entities(data, self.logger)
        self.logger.info(f"Entity Extraction is completed in {time.time() - start_time} seconds.")
        return entities

    def get_entities_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_entities(data))
        return x

    async def get_patient_information(self, data):
        """ This method is used to get phi dates from document """

        start_time = time.time()
        self.logger.info("Extraction of PHI and Document Type is started...")
        phi_and_doc_type_extractor = PHIAndDocTypeExtractor(self.logger)
        patient_information = await phi_and_doc_type_extractor.get_patient_information(data)
        self.logger.info(
            f"Extraction of PHI and Document Type is completed in {time.time() - start_time} seconds.")
        return patient_information

    def get_patient_information_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_patient_information(data))
        return x

    async def get_encounters(self, data, filename):
        """ This method is used to get phi dates from document """

        start_time = time.time()
        self.logger.info("Encounters Extraction is started...")
        encounters_extractor = EncountersExtractor(self.logger)
        encounter_events = await encounters_extractor.get_encounters(data, filename)
        self.logger.info(f"Encounters Extraction is completed in {time.time() - start_time} seconds.")
        return encounter_events

    def get_encounters_handler(self, data, filename):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_encounters(data, filename))
        return x

    async def process_doc(self, input_message):
        output_message = {'status': None, 'project_id': self.project_id, 'document_name': self.document_name,
                          'output_path': None}
        try:
            if input_message['Status'] != "SUCCEEDED":
                raise TextExtractionFailed

            page_wise_text = await self.textract_helper.get_page_wise_text(input_message)

            start_time = time.time()
            self.logger.info("Medical insights extraction started...")

            tasks = []
            with futures.ThreadPoolExecutor(2) as executor:
                tasks.append(executor.submit(self.get_summary_handler, data=page_wise_text))
                tasks.append(executor.submit(self.get_entities_handler, data=page_wise_text))
                tasks.append(
                    executor.submit(self.get_encounters_handler, data=page_wise_text, filename=self.document_name))
                tasks.append(executor.submit(self.get_patient_information_handler, data=page_wise_text))

            results = futures.wait(tasks)
            output = {}
            for res in results.done:
                output.update(res.result())

            s3_pdf_folder = os.path.dirname(input_message['DocumentLocation']['S3ObjectName'])
            s3_output_folder = s3_pdf_folder.replace(MedicalInsights.REQUEST_FOLDER_NAME,
                                                     MedicalInsights.RESPONSE_FOLDER_NAME)
            output_file_name = os.path.splitext(self.document_name)[0] + '_output.json'
            s3_output_key = os.path.join(s3_output_folder, output_file_name)
            output = json.dumps(output)
            output = output.encode("utf-8")
            await self.s3_utils.upload_object(AWS.S3.S3_BUCKET, s3_output_key, output)
            output_message['status'] = 'completed'
            output_message['output_path'] = s3_output_key
            self.logger.info(f"Medical insights extraction completed in {time.time() - start_time} seconds.")
        except Exception as e:
            output_message['status'] = 'failed'
            self.logger.error('%s -> %s' % (e, traceback.format_exc()))
        finally:
            self.logger.info(f'Publishing message: {json.dumps(output_message)} '
                             f'to Queue: {os.path.basename(AWS.SQS.LLM_OUTPUT_QUEUE)}')
            await self.sqs_helper.publish_message(AWS.SQS.LLM_OUTPUT_QUEUE, json.dumps(output_message))
