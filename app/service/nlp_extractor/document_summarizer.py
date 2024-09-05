import time

from fuzzywuzzy import fuzz
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client


class DocumentSummarizer:
    def __init__(self, logger):
        self.logger = logger
        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-3-haiku-20240307-v1:0'

        self.anthropic_llm = ChatBedrock(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 500,
                "stop_sequences": [],
            },
            client=self.bedrock_client,
        )
        self.reference_summary_first_last_line = MedicalInsights.LineRemove.SUMMARY_FIRST_LAST_LINE_REMOVE
        self.matching_threshold = 60

    async def __generate_summary(self, docs, query):
        qa = load_qa_chain(self.anthropic_llm, chain_type="stuff")
        chain_run = qa.invoke(input={'input_documents': docs, 'question': query})
        return chain_run['output_text']

    async def __data_formatter(self, json_data):
        """ This method is used to format the data and prepare chunks """

        raw_text = "".join(json_data.values())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(raw_text)

        chunk_length = []
        for text in texts:
            chunk_length.append(self.anthropic_llm.get_num_tokens(text))
        docs = [Document(page_content=t) for t in texts]
        return docs, chunk_length

    async def __first_line_remove(self, line, examples):
        words = line.split()
        start_of_first_line = ' '.join(words[:4])
        return any(
            fuzz.token_sort_ratio(start_of_first_line, example) > self.matching_threshold for example in examples)

    async def __last_line_remove(self, line, examples):
        words = line.split()
        start_of_last_line = ' '.join(words[:4])
        return any(fuzz.token_sort_ratio(start_of_last_line, example) > self.matching_threshold for example in examples)

    async def __get_stuff_calls(self, docs, chunk_length):
        stuff_calls = []
        current_chunk = []
        total_length = 0
        lengths = []

        for doc, length in zip(docs, chunk_length):
            if total_length + length >= 92000:
                lengths.append(total_length)

                stuff_calls.append(current_chunk)
                current_chunk = []
                total_length = 0
            current_chunk.append(doc)
            total_length += length

        if current_chunk:
            lengths.append(total_length)
            stuff_calls.append(current_chunk)

        return stuff_calls

    async def __post_processing(self, summary):
        """ This method is used to post-process the summary generated by LLM"""

        text = summary.replace('- ', '')
        text = text.strip()
        lines = text.split('\n')
        if await self.__first_line_remove(lines[0], self.reference_summary_first_last_line):
            lines = lines[1:]
        if await self.__last_line_remove(lines[-1], self.reference_summary_first_last_line):
            lines = lines[:-1]

        if lines[-1].__contains__('?'):
            lines = lines[:-1]

        modified_text = '\n'.join(lines)
        return modified_text.strip()

    async def get_summary(self, page_wise_text):
        """ This method is used to get the summary of document """

        json_data = page_wise_text
        raw_text = "".join(json_data.values()).strip()
        if not raw_text:
            return {"summary": MedicalInsights.TemplateResponse.SUMMARY_RESPONSE}
        chunk_prepare_start_time = time.time()
        docs, chunk_length = await self.__data_formatter(json_data)
        self.logger.info(f'[Summary] Chunk Preparation Time: {time.time() - chunk_prepare_start_time}')

        stuff_calls = await self.__get_stuff_calls(docs, chunk_length)

        query = MedicalInsights.Prompts.SUMMARY_PROMPT
        concatenate_query = MedicalInsights.Prompts.CONCATENATE_SUMMARY

        summary_start_time = time.time()
        summary = []

        if len(stuff_calls) == 1:
            if stuff_calls:
                docs = stuff_calls[0]
                summary = await self.__generate_summary(docs, query)

                input_tokens = (sum(chunk_length) + self.anthropic_llm.get_num_tokens(query))
                output_tokens = self.anthropic_llm.get_num_tokens(summary)

                self.logger.info(f'[Summary][{self.model_id_llm}] Input tokens: {input_tokens} '
                                 f'Output tokens: {output_tokens} LLM execution time: {time.time() - summary_start_time}')

        else:
            response_summary = [await self.__generate_summary(docs, query) for docs in stuff_calls]
            final_response_summary = [Document(page_content=response) for response in response_summary]
            summary = await self.__generate_summary(final_response_summary, concatenate_query)

            query_response = len(stuff_calls) * self.anthropic_llm.get_num_tokens(query)
            num_concatenate_query = self.anthropic_llm.get_num_tokens(concatenate_query)
            input_tokens = (sum(chunk_length) + query_response + num_concatenate_query)

            sum_response_summary = sum(self.anthropic_llm.get_num_tokens(rs) for rs in response_summary)
            output_tokens = sum_response_summary + self.anthropic_llm.get_num_tokens(summary)

            self.logger.info(f'[Summary][{self.model_id_llm}] Input tokens: {input_tokens} '
                             f'Output tokens: {output_tokens} LLM execution time: {time.time() - summary_start_time}')

        summary = await self.__post_processing(summary)
        return {"summary": summary}
