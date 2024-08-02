import asyncio
import json
import os
import time
from concurrent import futures

from app.common.sqs_helper import SQSHelper
from app.constant import AWS
from app.service.helper.textract_helper import get_page_wise_text
from app.service.nlp_extractor.document_summarizer import DocumentSummarizer
from app.service.nlp_extractor.encounters_extractor import EncountersExtractor
from app.service.nlp_extractor.entity_extractor import get_extracted_entities
from app.service.nlp_extractor.phi_and_doc_type_extractor import PHIAndDocTypeExtractor


class LLMProcessing:
    def __init__(self, logger, project_id, document_name):
        self.logger = logger
        self.project_id = project_id
        self.document_name = document_name
        self.sqs_helper = SQSHelper()

    async def get_summary(self, data):
        """ This method is used to get document summary """

        x = time.time()
        self.logger.info("[Medical-Insights] Summary generation is started...")
        document_summarizer = DocumentSummarizer(self.logger)
        summary = await document_summarizer.get_summary(data)
        self.logger.info(f"[Medical-Insights] Summary is generated in {time.time() - x} seconds.")
        return summary

    def get_summary_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_summary(data))
        return x

    async def get_entities(self, data):
        """ This method is used to get entities from document """

        x = time.time()
        self.logger.info("[Medical-Insights] Entity Extraction is started...")
        entities = await get_extracted_entities(data, self.logger)
        self.logger.info(f"[Medical-Insights] Entity Extraction is completed in {time.time() - x} seconds.")
        return entities

    def get_entities_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_entities(data))
        return x

    async def get_patient_information(self, data):
        """ This method is used to get phi dates from document """

        x = time.time()
        self.logger.info("[Medical-Insights] Extraction of PHI and Document Type is started...")
        phi_and_doc_type_extractor = PHIAndDocTypeExtractor(self.logger)
        patient_information = await phi_and_doc_type_extractor.get_patient_information(data)
        self.logger.info(
            f"[Medical-Insights] Extraction of PHI and Document Type is completed in {time.time() - x} seconds.")
        return patient_information

    def get_patient_information_handler(self, data):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_patient_information(data))
        return x

    async def get_encounters(self, data, filename):
        """ This method is used to get phi dates from document """

        x = time.time()
        self.logger.info("[Medical-Insights] Encounters Extraction is started...")
        encounters_extractor = EncountersExtractor(self.logger)
        encounter_events = await encounters_extractor.get_encounters(data, filename)
        self.logger.info(f"[Medical-Insights] Encounters Extraction is completed in {time.time() - x} seconds.")
        return encounter_events

    def get_encounters_handler(self, data, filename):
        _loop = asyncio.new_event_loop()
        x = _loop.run_until_complete(self.get_encounters(data, filename))
        return x

    async def publish_textract_failed_message(self):
        self.logger.warning('Textract failed!')
        message = {'status': 'failed', 'project_id': self.project_id, 'document_name': self.document_name,
                   'output_path': None}
        failed_message_body = json.dumps(message)
        self.logger.info(
            f'Publishing message: {failed_message_body} to Queue: {os.path.basename(AWS.SQS.LLM_OUTPUT_QUEUE)}')
        await self.sqs_helper.publish_message(AWS.SQS.LLM_OUTPUT_QUEUE, failed_message_body)

    async def process_doc(self, input_message):
        if input_message['Status'] != "SUCCEEDED":
            await self.publish_textract_failed_message()

        page_wise_text = await get_page_wise_text(input_message)

        tasks = []
        with futures.ThreadPoolExecutor(2) as executor:
            tasks.append(executor.submit(self.get_summary_handler, data=page_wise_text))
            tasks.append(executor.submit(self.get_entities_handler, data=page_wise_text))
            tasks.append(executor.submit(self.get_encounters_handler, data=page_wise_text, filename=self.document_name))
            tasks.append(executor.submit(self.get_patient_information_handler, data=page_wise_text))

        results = futures.wait(tasks)
        output = {}
        for x in results.done:
            output.update(x.result())

        print(output)
        return output
