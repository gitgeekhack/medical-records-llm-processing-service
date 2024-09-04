import json
import time
import re
import dateparser
from datetime import datetime, timedelta

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client


class PatientDemographicsExtractor:

    def __init__(self, logger):
        self.logger = logger
        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-3-haiku-20240307-v1:0'
        self.model_embeddings = 'amazon.titan-embed-text-v1'

        self.anthropic_llm = ChatBedrock(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0,
                "top_p": 0.01,
                "top_k": 1,
            },
            client=self.bedrock_client,
        )

        self.titan_llm = ChatBedrock(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)

    async def __parse_date(self, date):
        """ This method is used to parse the date into MM-DD-YYYY format """

        date = dateparser.parse(date, settings={'RELATIVE_BASE': datetime(1800, 1, 1)})
        if date and date.year != 1800:
            if date.year > datetime.now().year:
                date = date - timedelta(days=36525)
            date = date.strftime("%m-%d-%Y")
            return date
        return ""

    async def __extract_number(self, text):
        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            return match.group(1)
        return ""

    async def __process_patient_demographics(self, output_text):
        template_data = MedicalInsights.TemplateResponse.DEMOGRAPHICS_TEMPLATE_RESPONSE

        start_index = output_text.find('{')
        end_index = output_text.rfind('}') + 1
        json_str = output_text[start_index:end_index]

        if len(json_str) == 0:
            return {"patient_demographics": template_data}

        data = json.loads(json_str)

        # Handle date of birth
        date_of_birth = data.get('date_of_birth', '')
        if date_of_birth:
            x = await self.__parse_date(data['date_of_birth'])
            data['date_of_birth'] = x

        # Handle gender
        gender = data.get('gender', '')
        if gender:
            if data['gender'].lower().startswith('m'):
                data['gender'] = "Male"
            elif data['gender'].lower().startswith('f'):
                data['gender'] = "Female"
            else:
                data['gender'] = ""

        age = await self.__extract_number(data.get('age', ''))
        data['age'] = age

        height_data = data.get('height', {})
        if height_data:
            height_value = await self.__extract_number(height_data.get('value', ''))
            height_date = await self.__parse_date(height_data.get('date', ''))
            data['height']['value'], data['height']['date'] = height_value, height_date

        weight_data = data.get('weight', {})
        if weight_data:
            weight_value = await self.__extract_number(weight_data.get('value', ''))
            weight_date = await self.__parse_date(weight_data.get('date', ''))
            data['weight']['value'], data['weight'][
                'date'] = weight_value, weight_date

        final_data = {key: data.get(key, template_data[key]) for key in template_data}

        # Calculate BMI if height and weight are provided
        if final_data['height']['value'] and final_data['weight']['value']:
            bmi = round(703 * float(data['weight']['value']) / (
                        float(data['height']['value']) * float(data['height']['value'])), 2)
            final_data['bmi'] = str(bmi)
        else:
            final_data['bmi'] = ""

        return {"patient_demographics": final_data}

    async def __extract_patient_demographics(self, vectorstore_faiss):

        query = MedicalInsights.Prompts.PATIENT_DEMOGRAPHICS_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa.invoke({"query": query})
        response = answer['result']
        final_response = await self.__process_patient_demographics(response)
        return final_response

    async def __data_formatter(self, page_wise_text):
        raw_text = ""
        raw_text += " ".join(page_wise_text.values())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)
        for text in texts:
            threshold = self.anthropic_llm.get_num_tokens(text)
            if threshold > 7000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=15000, chunk_overlap=200
                )
                texts = text_splitter.split_text(raw_text)
                break

        docs = [Document(page_content=t) for t in texts]
        return docs

    async def __get_docs_embeddings(self, page_wise_text):
        """ This method is used to prepare the embeddings and returns it """

        x = time.time()
        docs = await self.__data_formatter(page_wise_text)

        self.logger.info(f'[Demographics] Chunk Preparation Time: {time.time() - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )
        return vector_embeddings


    async def get_patient_demographics(self, page_wise_text):
        """ This is expose method of the class """

        pdf_text = "".join(page_wise_text.values()).strip()
        if not pdf_text:
            template_data = MedicalInsights.TemplateResponse.DEMOGRAPHICS_TEMPLATE_RESPONSE
            return {"patient_demographics": template_data}
        else:
            t = time.time()
            embeddings = await self.__get_docs_embeddings(page_wise_text)
            self.logger.info(
                f"[Demographics] Embedding Generation time: {time.time() - t}")

            t = time.time()
            patient_demographics = await self.__extract_patient_demographics(embeddings)
            self.logger.info(f'[Demographics][{self.model_id_llm}] LLM execution time: {time.time() - t}')

            return {"patient_demographics": patient_demographics['patient_demographics']}


