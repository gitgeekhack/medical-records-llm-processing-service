import os
import re
import time
import json
import dateparser
from datetime import datetime, timedelta

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import logger
from app.constant import AWS
from app.constant import MedicalInsights
from app.service.medical_document_insights.nlp_extractor import bedrock_client, get_llm_input_tokens


class PHIAndDocTypeExtractor:
    def __init__(self):
        os.environ['AWS_DEFAULT_REGION'] = AWS.BotoClient.AWS_DEFAULT_REGION

        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-instant-v1'
        self.model_embeddings = 'amazon.titan-embed-text-v1'

        self.anthropic_llm = Bedrock(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens_to_sample": 4000,
                "temperature": 0.75,
                "top_p": 0.01,
                "top_k": 0,
                "stop_sequences": [],
            },
            client=self.bedrock_client,
        )

        self.titan_llm = Bedrock(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)

    async def __get_docs_embeddings(self, data):
        """ This method is used to prepare the embeddings and returns it """

        x = time.time()
        docs = await self.__data_formatter(data)

        emb_tokens = 0
        for i in docs:
            emb_tokens += self.titan_llm.get_num_tokens(i.page_content)

        y = time.time()
        logger.info(f'[Medical-Insights][PHI-Embeddings] Chunk Preparation Time: {y - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )
        logger.info(f'[Medical-Insights][PHI-Embeddings][{self.model_embeddings}] Input embedding tokens: {emb_tokens}'
                    f'and Generation time: {time.time() - y}')

        return vector_embeddings

    async def __get_key(self, key):
        """ This method is used to provide the JSON key """

        if "injury" in key.lower():
            result_key = "injury_dates"
        elif "admission" in key.lower():
            result_key = "admission_dates"
        elif "discharge" in key.lower():
            result_key = "discharge_dates"

        return result_key

    async def __data_formatter(self, json_data):
        """ This method is used to format the data and prepare chunks """

        raw_text = "".join(json_data.values())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000, chunk_overlap=200
        )

        texts = text_splitter.split_text(raw_text)

        for text in texts:
            threshold = self.anthropic_llm.get_num_tokens(text)
            if threshold > 5000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=200
                )
                texts = text_splitter.split_text(raw_text)
                break

        docs = [Document(page_content=t) for t in texts]
        return docs

    async def __extract_values_between_curly_braces(self, text):

        pattern = r'\{.*?\}'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    async def __parse_date(self, date):
        """ This method is used to parse the date into MM-DD-YYYY format """

        date = dateparser.parse(date, settings={'RELATIVE_BASE': datetime(1800, 1, 1)})
        if date and date.year != 1800:
            if date.year > datetime.now().year:
                date = date - timedelta(days=36525)
            date = date.strftime("%m-%d-%Y")
            return date
        return "None"

    async def __process_patient_name_and_dob(self, text):
        """ This method is used to convert the string response of LLM into the JSON """

        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        json_str = text[start_index:end_index]

        final_data = {'patient_name': 'None', 'date_of_birth': 'None'}
        if not json_str or not eval(json_str):
            return final_data

        data = json.loads(json_str)
        data_keys = ['patient_name', 'date_of_birth']

        updated_final_data = dict(zip(data_keys, list(data.values())))
        for key in data_keys:
            if key not in updated_final_data.keys():
                updated_final_data[key] = 'None'

        if updated_final_data['date_of_birth'] and isinstance(updated_final_data['date_of_birth'], str):
            x = await self.__parse_date(updated_final_data['date_of_birth'])
            updated_final_data['date_of_birth'] = x

        return updated_final_data

    async def __get_document_type(self, embeddings):
        """ This method is used to get the document type using vectored embeddings """

        doc_type_query = MedicalInsights.Prompts.DOC_TYPE_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        logger.info(f'[Medical-Insights][PHI-DocumentType][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(doc_type_query) + self.titan_llm.get_num_tokens(prompt_template)}')

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=embeddings.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa({"query": doc_type_query})

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.anthropic_llm.get_num_tokens(prompt_template)
        output_tokens = self.anthropic_llm.get_num_tokens(answer['result'])

        logger.info(f'[Medical-Insights][PHI-DocumentType][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens}')

        response = json.loads(answer['result'])
        doc_type_value = response['document_type']
        return doc_type_value

    async def __parse_dates_in_phi_response(self, response):
        parsed_response = {}
        for key, dates in response.items():
            parsed_dates = []
            for date in dates:
                parsed_date = await self.__parse_date(date)
                parsed_dates.append(parsed_date)
            parsed_response[key] = parsed_dates
        return parsed_response

    async def __get_phi_dates(self, embeddings, document_type):
        """ This method is to provide the PHI dates from the document """

        query = MedicalInsights.Prompts.PHI_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        logger.info(f'[Medical-Insights][PHI-Dates][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(query) + self.titan_llm.get_num_tokens(prompt_template)}')

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=embeddings.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa({"query": query})
        response = answer['result']

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.anthropic_llm.get_num_tokens(prompt_template)
        output_tokens = self.anthropic_llm.get_num_tokens(response)

        logger.info(f'[Medical-Insights][PHI-Dates][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens}')

        matches = await self.__extract_values_between_curly_braces(response)
        json_result = json.loads(matches[0])
        dates = {}

        for key, value in json_result.items():
            result_key = await self.__get_key(key)
            dates[result_key] = value if isinstance(value, list) else [value]

        parsed_dates = await self.__parse_dates_in_phi_response(dates)

        if document_type == "Ambulance" or document_type == "Emergency":
            dates["injury_dates"] = dates["admission_dates"]

        return {'patient_information': parsed_dates}

    async def __get_patient_name_and_dob(self, embeddings):
        """ This method is to provide the Patient Name and DOB from the document """

        query = MedicalInsights.Prompts.PATIENT_INFO_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        logger.info(f'[Medical-Insights][PHI-Name&DOB][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(query) + self.titan_llm.get_num_tokens(prompt_template)}')

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=embeddings.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa({"query": query})

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.anthropic_llm.get_num_tokens(prompt_template)
        output_tokens = self.anthropic_llm.get_num_tokens(answer['result'])

        logger.info(f'[Medical-Insights][PHI-Name&DOB][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens}')

        processed_result = await self.__process_patient_name_and_dob(answer['result'])
        return processed_result

    async def get_patient_information(self, data):
        """ This is expose method of the class """

        raw_text = "".join(data.values()).strip()
        if not raw_text:
            return MedicalInsights.TemplateResponse.PHI_RESPONSE

        t = time.time()
        embeddings = await self.__get_docs_embeddings(data)
        logger.info(f"[Medical-Insights][PHI] Embedding Generation for PHI and Document Type is completed in {time.time() - t} seconds.")

        t = time.time()
        document_type = await self.__get_document_type(embeddings)
        logger.info(f"[Medical-Insights][PHI] Identification of Document Type is completed in {time.time() - t} seconds.")

        t = time.time()
        patient_info = await self.__get_phi_dates(embeddings, document_type)
        logger.info(f"[Medical-Insights][PHI] PHI Dates Extraction is completed in {time.time() - t} seconds.")

        t = time.time()
        patient_name_and_dob = await self.__get_patient_name_and_dob(embeddings)
        logger.info(f"[Medical-Insights][PHI] Patient Name and DOB Extraction is completed in {time.time() - t} seconds.")

        patient_info['patient_information'].update(patient_name_and_dob)
        return patient_info
