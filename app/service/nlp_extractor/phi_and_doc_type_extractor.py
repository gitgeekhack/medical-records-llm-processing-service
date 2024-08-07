import json
import re
import time
from datetime import datetime, timedelta

import dateparser
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockLLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client


class PHIAndDocTypeExtractor:
    def __init__(self, logger):
        self.logger = logger
        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-instant-v1'
        self.model_embeddings = 'amazon.titan-embed-text-v1'

        self.anthropic_llm = BedrockLLM(
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

        self.titan_llm = BedrockLLM(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)

    async def __get_docs_embeddings(self, data):
        """ This method is used to prepare the embeddings and returns it """

        chunk_start_time = time.time()
        docs = await self.__data_formatter(data)

        emb_start_time = time.time()
        self.logger.info(f'[PHI-Embeddings] Chunk Preparation Time: {emb_start_time - chunk_start_time}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )
        self.logger.info(f'[PHI-Embeddings][{self.model_embeddings}]'
                         f' Embeddings generation time: {time.time() - emb_start_time}')

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

        answer = qa.invoke({"query": doc_type_query})

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

        answer = qa.invoke({"query": query})
        response = answer['result']

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

        answer = qa.invoke({"query": query})

        processed_result = await self.__process_patient_name_and_dob(answer['result'])
        return processed_result

    async def get_patient_information(self, data):
        """ This is expose method of the class """

        raw_text = "".join(data.values()).strip()
        if not raw_text:
            return MedicalInsights.TemplateResponse.PHI_RESPONSE

        emb_start_time = time.time()
        embeddings = await self.__get_docs_embeddings(data)
        self.logger.info(
            f"[PHI] Embedding Generation for PHI and Document Type is completed in {time.time() - emb_start_time} seconds.")

        doctype_start_time = time.time()
        document_type = await self.__get_document_type(embeddings)
        self.logger.info(
            f"[PHI] Identification of Document Type is completed in {time.time() - doctype_start_time} seconds.")

        dates_start_time = time.time()
        patient_info = await self.__get_phi_dates(embeddings, document_type)
        self.logger.info(f"[PHI] PHI Dates Extraction is completed in {time.time() - dates_start_time} seconds.")

        name_dob_start_time = time.time()
        patient_name_and_dob = await self.__get_patient_name_and_dob(embeddings)
        self.logger.info(
            f"[PHI] Patient Name and DOB Extraction is completed in {time.time() - name_dob_start_time} seconds.")

        patient_info['patient_information'].update(patient_name_and_dob)
        return patient_info
