import copy
import os
import time
import json
import shutil
import aiofiles

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import logger
from app.common.s3_utils import s3_utils
from app.constant import AWS, MedicalInsights
from app.service.medical_document_insights.nlp_extractor import bedrock_client, get_llm_input_tokens


class DocumentQnA:
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
        self.prompt = self.__create_prompt_template()

    def __create_prompt_template(self):
        prompt_template = """
        Human: You are a Medical Assistant that provides concise answers to the questions related to the medical text context given to you. Strictly answer the questions related to the following information 
        <context>
        {context}
        </context>
        to answer in a helpful manner. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Question: {question}

        Medical Assistant:"""

        self.prompt_template_tokens = self.anthropic_llm.get_num_tokens(prompt_template)

        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        return prompt

    async def __prepare_data(self, project_path):

        s3_embedding_path = project_path.replace(MedicalInsights.REQUEST_FOLDER_NAME,
                                                 MedicalInsights.EMBEDDING_FOLDER_NAME)

        embeddings_pickle_file_path = os.path.join(s3_embedding_path, MedicalInsights.EMBEDDING_PICKLE_FILE_NAME)
        response = await s3_utils.check_s3_path_exists(bucket=AWS.S3.MEDICAL_BUCKET_NAME,
                                                       key=embeddings_pickle_file_path)

        local_embedding_pickle_path = embeddings_pickle_file_path.replace(MedicalInsights.S3_FOLDER_NAME,
                                                                          MedicalInsights.LOCAL_FOLDER_NAME)
        local_embedding_faiss_path = local_embedding_pickle_path.replace(MedicalInsights.EMBEDDING_PICKLE_FILE_NAME,
                                                                         MedicalInsights.EMBEDDING_FAISS_FILE_NAME)
        local_embedding_dir = os.path.dirname(local_embedding_pickle_path)
        os.makedirs(local_embedding_dir, exist_ok=True)

        if response:
            s3_embedding_pickle_path = local_embedding_pickle_path.replace(MedicalInsights.LOCAL_FOLDER_NAME,
                                                                           MedicalInsights.S3_FOLDER_NAME)
            s3_embedding_faiss_path = local_embedding_faiss_path.replace(MedicalInsights.LOCAL_FOLDER_NAME,
                                                                         MedicalInsights.S3_FOLDER_NAME)

            await s3_utils.download_object(AWS.S3.MEDICAL_BUCKET_NAME, s3_embedding_pickle_path,
                                           local_embedding_pickle_path,
                                           AWS.S3.ENCRYPTION_KEY)
            await s3_utils.download_object(AWS.S3.MEDICAL_BUCKET_NAME, s3_embedding_faiss_path,
                                           local_embedding_faiss_path,
                                           AWS.S3.ENCRYPTION_KEY)
            vectored_data = FAISS.load_local(local_embedding_dir, self.bedrock_embeddings, index_name='embeddings',
                                             allow_dangerous_deserialization=True)

        else:

            s3_textract_paths = project_path.replace(MedicalInsights.REQUEST_FOLDER_NAME,
                                                     MedicalInsights.TEXTRACT_FOLDER_NAME)
            local_textract_paths = s3_textract_paths.replace(MedicalInsights.S3_FOLDER_NAME,
                                                             MedicalInsights.LOCAL_FOLDER_NAME)
            os.makedirs(local_textract_paths, exist_ok=True)

            response = await s3_utils.check_s3_path_exists(bucket=AWS.S3.MEDICAL_BUCKET_NAME, key=s3_textract_paths)

            raw_text = ""
            for item in response['Contents']:

                if item['Key'].endswith('.json'):
                    file_name = os.path.basename(item['Key'])
                    local_textract_file_path = os.path.join(local_textract_paths, file_name)
                    await s3_utils.download_object(AWS.S3.MEDICAL_BUCKET_NAME, item['Key'],
                                                   local_textract_file_path, AWS.S3.ENCRYPTION_KEY)
                    with open(local_textract_file_path, 'r') as file:
                        data = json.loads(file.read())
                        raw_text = raw_text + "".join(data.values())

            if len(raw_text.strip()) != 0:
                docs = await self.__data_formatter(raw_text)
                vectored_data = await self.__prepare_embeddings(docs, s3_embedding_path)
            else:
                vectored_data = None

        return vectored_data

    async def __data_formatter(self, raw_text):
        """ This method is used to format the data and prepare chunks """

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

    async def __prepare_embeddings(self, docs, s3_embedding_path):
        x = time.time()
        emb_tokens = 0
        for i in docs:
            emb_tokens += self.titan_llm.get_num_tokens(i.page_content)

        y = time.time()
        logger.info(f'[Medical-Insights][QnA-Embeddings] Chunk Preparation Time: {y - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )

        local_embedding_path = s3_embedding_path.replace(MedicalInsights.S3_FOLDER_NAME,
                                                         MedicalInsights.LOCAL_FOLDER_NAME)
        os.makedirs(local_embedding_path, exist_ok=True)

        vector_embeddings.save_local(local_embedding_path, index_name='embeddings')

        s3_embedding_pickle_path = os.path.join(s3_embedding_path, MedicalInsights.EMBEDDING_PICKLE_FILE_NAME)
        s3_embedding_vector_path = os.path.join(s3_embedding_path, MedicalInsights.EMBEDDING_FAISS_FILE_NAME)

        async with aiofiles.open(os.path.join(local_embedding_path, MedicalInsights.EMBEDDING_PICKLE_FILE_NAME),
                                 'rb') as f:
            file_obj = await f.read()
        await s3_utils.upload_object(AWS.S3.MEDICAL_BUCKET_NAME, s3_embedding_pickle_path, file_obj,
                                     AWS.S3.ENCRYPTION_KEY)

        async with aiofiles.open(os.path.join(local_embedding_path, MedicalInsights.EMBEDDING_FAISS_FILE_NAME),
                                 'rb') as f:
            file_obj = await f.read()
        await s3_utils.upload_object(AWS.S3.MEDICAL_BUCKET_NAME, s3_embedding_vector_path, file_obj,
                                     AWS.S3.ENCRYPTION_KEY)

        logger.info(
            f'[Medical-Insights][QnA-Embeddings][{self.model_embeddings}] Input embedding tokens: {emb_tokens}'
            f'and Generation time: {time.time() - y}')
        return vector_embeddings

    async def __create_conversation_chain(self, vectored_data, prompt_template):

        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectored_data.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

        return qa

    async def get_query_response(self, query, project_path):

        x = time.time()
        vectored_data = await self.__prepare_data(project_path)
        logger.info(f"[Medical-Insights-QnA] Input data preparation for LLM is completed in {time.time() - x} seconds.")

        if vectored_data is None:
            logger.warning("[Medical-Insights-QnA] Empty Document Found for QnA !!")
            response = copy.deepcopy(MedicalInsights.TemplateResponse.QNA_RESPONSE)
            response['query'] = query
            return response

        x = time.time()
        conversation_chain = await self.__create_conversation_chain(vectored_data, self.prompt)
        answer = conversation_chain({'query': query})

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.prompt_template_tokens
        output_tokens = self.anthropic_llm.get_num_tokens(answer['result'])

        local_path = project_path.replace(MedicalInsights.S3_FOLDER_NAME, MedicalInsights.LOCAL_FOLDER_NAME)
        project_id = os.path.dirname(local_path[:-1])
        shutil.rmtree(project_id)

        logger.info(f'[Medical-Insights-QnA][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(query) + self.prompt_template_tokens}')

        logger.info(f'[Medical-Insights-QnA][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens} LLM execution time: {time.time() - x}')

        logger.info(f"[Medical-Insights-QnA] LLM generated response for input query in {time.time() - x} seconds.")

        return answer
