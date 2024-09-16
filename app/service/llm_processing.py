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
from app.service.nlp_extractor.medical_chronology_extractor import MedicalChronologyExtractor
from app.service.nlp_extractor.entity_extractor import get_extracted_entities
from app.service.nlp_extractor.patient_demographics_extractor import PatientDemographicsExtractor
from app.service.nlp_extractor.doc_type_extractor import DocTypeExtractor
from app.service.nlp_extractor.history_extractor import  HistoryExtractor

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

    async def get_patient_demographics(self, data):
        """ This method is used to get patient demographics from document """

        x = time.time()
        self.logger.info("Patient Demographics Extraction is started...")
        demographics_extractor = PatientDemographicsExtractor(self.logger)
        patient_demographics = await demographics_extractor.get_patient_demographics(data)
        self.logger.info(f"Patient Demographics Extraction is completed in {time.time() - x} seconds.")
        return patient_demographics

    def get_patient_demographics_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_patient_demographics(data))
        return x

    async def get_document_type(self, data):
        """ This method is used to get document type from document """

        x = time.time()
        self.logger.info("Document Type Extraction is started...")
        doc_type_extractor = DocTypeExtractor(self.logger)
        doc_type = await doc_type_extractor.extract_document_type(data)
        self.logger.info(f"Document Type Extraction is completed in {time.time() - x} seconds.")
        return doc_type

    def get_document_type_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_document_type(data))
        return x

    async def get_chronology(self, data, filename):
        """ This method is used to get medical chronology from document """

        start_time = time.time()
        self.logger.info("Medical Chronology Extraction is started...")
        encounters_extractor = MedicalChronologyExtractor(self.logger)
        encounter_events = await encounters_extractor.get_medical_chronology(data, filename)
        self.logger.info(f"Medical Chronology Extraction is completed in {time.time() - start_time} seconds.")
        return encounter_events

    def get_chronology_handler(self, data, filename):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_chronology(data, filename))
        return x

    async def get_history(self, data):
        """ This method is used to get History and Psychiatric Injury from document """

        x = time.time()
        self.logger.info("History & Psychiatric Injury Extraction is started...")
        history_extractor = HistoryExtractor(self.logger)
        history_info = await history_extractor.get_history(data)
        self.logger.info(
            f"History & Psychiatric Injury Extraction is completed in {time.time() - x} seconds.")
        return history_info

    def get_history_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_history(data))
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
                    executor.submit(self.get_chronology_handler, data=page_wise_text, filename=self.document_name))
                tasks.append(executor.submit(self.get_patient_demographics_handler, data=page_wise_text))
                tasks.append(executor.submit(self.get_history_handler, data=page_wise_text))
                tasks.append(executor.submit(self.get_document_type_handler, data=page_wise_text))


            results = futures.wait(tasks)
            output = {}
            for res in results.done:
                output.update(res.result())

            output['tests'] = []
            for entity in output['medical_entities']:
                tests = entity.pop('tests')
                if tests:
                    output['tests'].append({
                        'page_no': entity['page_no'],
                        'tests': tests
                    })
            output['document_name'] = self.document_name

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
                             f'to Queue: {os.path.basename(AWS.SQS.LLM_OUTPUT_QUEUE_URL)}')
            await self.sqs_helper.publish_message(AWS.SQS.LLM_OUTPUT_QUEUE_URL, json.dumps(output_message))
