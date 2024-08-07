import asyncio
import json
import re
import traceback
from concurrent import futures

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockLLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client

model_id_llm = 'anthropic.claude-instant-v1'
model_embeddings = 'amazon.titan-embed-text-v1'

anthropic_llm = BedrockLLM(
    model_id=model_id_llm,
    model_kwargs={
        "max_tokens_to_sample": 10000,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 250
    },
    client=bedrock_client,
)

titan_llm = BedrockLLM(model_id=model_embeddings, client=bedrock_client)
bedrock_embeddings = BedrockEmbeddings(model_id=model_embeddings, client=bedrock_client)

logger = None


async def is_alpha(entity):
    """ This method is used to validate entity by checking alphabet is present or not """

    pattern = r'[a-zA-Z]'

    if re.search(pattern, entity):
        return True
    else:
        return False


async def get_valid_entity(entities):
    """ This method is used to validate entity by checking if alphabetic character is present or not """

    valid_entities = {key: [] for key in entities}

    for key, entity_list in entities.items():
        for entity in entity_list:
            processed_entity = await is_alpha(entity)
            if processed_entity and entity.strip():
                x = entity[0].upper() + entity[1:]
                valid_entities[key].append(x)
    return valid_entities


async def convert_str_into_json(text):
    """ This method is used to convert str into json object with consistent key-name """
    global logger
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    json_str = text[start_index:end_index]
    final_data = {'diagnosis': [], 'treatments': [], 'medications': []}

    if not json_str or not eval(json_str):
        return final_data

    try:
        data = json.loads(json_str)
        data_keys = ['diagnosis', 'treatments', 'medications']
        updated_final_data = dict(zip(data_keys, list(data.values())))
        for key in data_keys:
            if key not in updated_final_data.keys():
                updated_final_data[key] = []
        final_data = await get_valid_entity(updated_final_data)

    except Exception as e:
        logger.error('%s -> %s' % (e, traceback.format_exc()))
        return final_data

    return final_data


async def data_formatter(json_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100, chunk_overlap=200
    )
    texts = text_splitter.split_text(json_data)
    docs = [Document(page_content=t) for t in texts]
    return docs


async def get_medical_entities(key, value, page_wise_entities):
    """ This method is used to provide medical entities """

    docs = await data_formatter(value)

    if not docs:
        page_wise_entities[key] = {'diagnosis': [], 'treatments': [], 'medications': []}
        return page_wise_entities

    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=bedrock_embeddings,
    )

    query = MedicalInsights.Prompts.ENTITY_PROMPT
    prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=anthropic_llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    answer = qa.invoke({"query": query})
    response = answer['result']

    entities = await convert_str_into_json(response)
    page_wise_entities[key] = entities
    return page_wise_entities


def extract_entity_handler(key, value, page_wise_entities):
    _loop = asyncio.new_event_loop()
    x = _loop.run_until_complete(get_medical_entities(key, value, page_wise_entities))
    return x


async def get_extracted_entities(json_data, logger_instance):
    """ This method is used to provide medical entities from document"""

    global logger
    logger = logger_instance

    entity = {}

    task = []
    with futures.ThreadPoolExecutor(2) as executor:
        for key, value in json_data.items():
            new_future = executor.submit(extract_entity_handler, key=key,
                                         value=value, page_wise_entities=entity)
            task.append(new_future)

    results = futures.wait(task)

    page_wise_entities = {}
    for entity in results.done:
        page_wise_entities.update(entity.result())

    filter_empty_pages = {key: value for key, value in page_wise_entities.items() if
                          any(value[key] for key in ['diagnosis', 'treatments', 'medications'])}
    sorted_filtered = dict(sorted(filter_empty_pages.items(), key=lambda item: int(item[0].split('_')[1])))
    page_wise_entities = []
    for page_k, page_v in sorted_filtered.items():
        page_no = int(page_k.split("_")[1])
        page_v |= {"page_no": page_no}
        page_wise_entities.append(page_v)

    return {'medical_entities': page_wise_entities}
