import asyncio
import json
import re
import traceback
from concurrent import futures
import dateparser
import pytz
from datetime import datetime

from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings

from app.constant import MedicalInsights
from app.service.nlp_extractor import bedrock_client

model_id_llm = 'anthropic.claude-3-sonnet-20240229-v1:0'
model_embeddings = 'amazon.titan-embed-text-v1'

anthropic_llm = ChatBedrock(
    model_id=model_id_llm,
    model_kwargs={
        "max_tokens": 10000,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 250
    },
    client=bedrock_client,
)

titan_llm = ChatBedrock(model_id=model_embeddings, client=bedrock_client)
bedrock_embeddings = BedrockEmbeddings(model_id=model_embeddings, client=bedrock_client)

logger = None


async def is_alpha(entity):
    """ This method is used to validate entity by checking alphabet is present or not """

    pattern = r'[a-zA-Z]'

    if re.search(pattern, entity):
        return True
    else:
        return False

async def parse_date(date):
    """ This method is used to parse the date in the datetime format """

    date = dateparser.parse(date, settings={'RELATIVE_BASE': datetime(1800, 1, 1)})
    if date:
        date = date.replace(tzinfo=pytz.UTC)
        return date.strftime('%m-%d-%Y')
    return None

async def get_valid_entity(entities, page_number):
    """ This method is used to validate entities by checking if alphabetic character is present or not """

    valid_entities = {
        "diagnosis": {"allergies": [], "pmh": [], "others": []},
        "treatments": {"pmh": [], "others": []},
        "tests": [],
        "medications": {"pmh": [], "others": []},
        "page_no": page_number
    }

    async def validate_entity_field(entity):
        """ Validates if alphabetic character is present or not """

        processed_entity = await is_alpha(entity)
        if processed_entity and entity.strip():
            return entity[0].upper() + entity[1:]
        else:
            return ""

    for category, subcategories in entities.items():
        if category == "medications":
            for subcategory, entity_list in subcategories.items():
                for index, entity in enumerate(entity_list):
                    valid_name = await validate_entity_field(entity.get("name", None))
                    valid_dosage = await validate_entity_field(entity.get("dosage", None))
                    if valid_name:
                        entity['name'] = valid_name
                        if valid_dosage:
                            entity['dosage'] = valid_dosage
                        valid_entities[category][subcategory].append(entity)

        elif category == "tests":
            for entity_dic in subcategories:
                valid_name = await validate_entity_field(entity_dic.get("name", None))
                valid_date = await parse_date(entity_dic.get("date", None))
                if valid_name:
                    entity_dic['name'] = valid_name
                    if valid_date:
                        entity_dic['date'] = valid_date
                    valid_entities[category].append(entity_dic)

        elif category in ['diagnosis', 'treatments']:
            for subcategory, entity_list in subcategories.items():
                for index, entity in enumerate(entity_list):
                    valid_entity = await validate_entity_field(entity)
                    if valid_entity:
                        valid_entities[category][subcategory].append(valid_entity)

    return valid_entities


async def convert_str_into_json(text, page_number):
    """ This method is used to convert str into json object with consistent key-name """

    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    json_str = text[start_index:end_index]

    final_data = {
        "diagnosis": {"allergies": [], "pmh": [], "others": []},
        "treatments": {"pmh": [], "others": []},
        "tests": [],
        "medications": {"pmh": [], "others": []},
        "page_no": page_number
    }

    if not json_str or not eval(json_str):
        return final_data

    try:
        data = json.loads(json_str)
        final_data = await get_valid_entity(data, page_number)

    except Exception as e:
        logger.error('%s -> %s', e, traceback.format_exc())
        return final_data

    return final_data


async def data_formatter(json_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=200
    )
    texts = text_splitter.split_text(json_data)
    docs = [Document(page_content=t) for t in texts]

    return docs

async def get_page_number(key):
    pattern = r'page_(\d+)'

    match = re.search(pattern, key)

    if match:
        page_number = int(match.group(1))
    else:
        page_number = 1

    return page_number

async def get_medical_entities(key, value):
    """ This method is used to provide medical entities """

    docs = await data_formatter(value)

    page_number = await get_page_number(key)

    if not docs:
        return {
            "diagnosis": {"allergies": [], "pmh": [], "others": []},
            "treatments": {"pmh": [], "others": []},
            "tests": [],
            "medications": {"pmh": [], "others": []},
            "page_no": page_number
        }

    diagnosis_treatment_query = MedicalInsights.Prompts.DIAGNOSIS_TREATMENT_ENTITY_PROMPT
    tests_medication_query = MedicalInsights.Prompts.TESTS_MEDICATION_ENTITY_PROMPT

    chain_qa = load_qa_chain(anthropic_llm, chain_type="stuff")
    diagnosis_treatment_result = chain_qa.invoke(input={"input_documents": docs, "question": diagnosis_treatment_query})
    tests_medication_result = chain_qa.invoke(input={"input_documents": docs, "question": tests_medication_query})

    page_entities = await convert_str_into_json(diagnosis_treatment_result['output_text'], page_number)
    tests_medication_entities = await convert_str_into_json(tests_medication_result['output_text'], page_number)
    page_entities['tests'] = tests_medication_entities['tests']
    page_entities['medications'] = tests_medication_entities['medications']

    page_entities['page_no'] = page_number
    return page_entities


def extract_entity_handler(key, value):
    _loop = asyncio.new_event_loop()
    x = _loop.run_until_complete(get_medical_entities(key, value))
    return x

async def get_extracted_entities(page_wise_text, logger_instance):
    """ This method is used to provide medical entities from document"""

    global logger
    logger = logger_instance

    json_data = page_wise_text

    task = []
    with futures.ThreadPoolExecutor(2) as executor:
        for key, value in json_data.items():
            new_future = executor.submit(extract_entity_handler, key=key, value=value)
            task.append(new_future)

    results = futures.wait(task)

    page_wise_entities = []
    for entity in results.done:
        page_wise_entities.append(entity.result())

    filtered_empty_pages = [page for page in page_wise_entities if
                            any(any(sub_dict.values()) for sub_dict in page.values() if isinstance(sub_dict, dict))]

    sorted_page_wise_entities = sorted(filtered_empty_pages, key=lambda k: k['page_no'])

    return {'medical_entities': sorted_page_wise_entities}
