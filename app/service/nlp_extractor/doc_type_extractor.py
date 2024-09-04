import json
import time

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client


class DocTypeExtractor:

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

    async def __process_document_type(self, output_text):

        start_index = output_text.find('{')
        end_index = output_text.rfind('}') + 1
        json_str = output_text[start_index:end_index]

        if len(json_str) == 0:
            return {"document_type": ""}

        data = json.loads(json_str)
        return {"document_type": data.get("document_type", '')}

    async def __classify_document_type(self, vectorstore_faiss):

        query = MedicalInsights.Prompts.DOC_TYPE_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa.invoke({"query": query})
        response = answer['result']
        final_response = await self.__process_document_type(response)
        return final_response

    async def __data_formatter(self, page_wise_text):
        json_data = page_wise_text
        raw_text = "".join(json_data.values())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)
        for text in texts:
            threshold = self.anthropic_llm.get_num_tokens(text)
            if threshold > 7000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=200
                )
                texts = text_splitter.split_text(raw_text)
                break
        docs = [Document(page_content=t) for t in texts]
        return docs

    async def __get_docs_embeddings(self, page_wise_text):
        """ This method is used to prepare the embeddings and returns it """

        x = time.time()
        docs = await self.__data_formatter(page_wise_text)

        y = time.time()
        self.logger.info(f'[Document-Type] Chunk Preparation Time: {y - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )

        return vector_embeddings


    async def extract_document_type(self, page_wise_text):
        """ This is expose method of the class """

        pdf_text = "".join(page_wise_text.values()).strip()
        if not pdf_text:
            return {"document_type": ""}
        else:
            emb_start_time = time.time()
            embeddings = await self.__get_docs_embeddings(page_wise_text)
            self.logger.info(f"[Document-Type] Embedding Generation time: {time.time() - emb_start_time}")

            document_type_start_time = time.time()
            document_type = await self.__classify_document_type(embeddings)
            self.logger.info(f"[Document-Type] LLM Execution time: {time.time() - document_type_start_time}")

            return document_type



