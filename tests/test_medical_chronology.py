import json

import pytest
import logging

from langsmith import trace
from sqlalchemy.util import await_only

from app.service.nlp_extractor.medical_chronology_extractor import MedicalChronologyExtractor
from langchain.docstore.document import Document

pytest_plugins = ('pytest_asyncio',)

class TestMedicalChronologyExtractor:
    @pytest.mark.asyncio
    async def test_find_page_range_with_valid_parameters(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        page_indexes_dict = {'page_1': [0, 2000], 'page_2': [2001, 4000],'page_3': [4001, 6000]}
        chunk_indexes = [3000, 4900]
        search_start_page = 1
        page_range = await enc._MedicalChronologyExtractor__find_page_range(page_indexes_dict, chunk_indexes, search_start_page)
        assert page_range == (2, 3, 3)

    @pytest.mark.asyncio
    async def test_find_page_range_with_invalid_search_start_page(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        page_indexes_dict = {'page_1': [0, 2000], 'page_2': [2001, 4000],'page_3': [4001, 6000]}
        chunk_indexes = [3000, 4900]
        search_start_page = 4
        page_range = await enc._MedicalChronologyExtractor__find_page_range(page_indexes_dict, chunk_indexes, search_start_page)
        assert page_range == (None, None, 4)

    @pytest.mark.asyncio
    async def test_find_page_range_with_invalid_chunk_indexes(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        page_indexes_dict = {'page_1': [0, 2000], 'page_2': [2001, 4000],'page_3': [4001, 6000]}
        chunk_indexes = [6500, 7900]
        search_start_page = 1
        page_range = await enc._MedicalChronologyExtractor__find_page_range(page_indexes_dict, chunk_indexes, search_start_page)
        assert page_range == (None, None, 1)

    @pytest.mark.asyncio
    async def test_find_page_range_with_invalid_page_indexes(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        page_indexes_dict = {'page_1': [-100, -2000], 'page_2': [-2001, -4000],'page_3': [-4001, -6000]}
        chunk_indexes = [3000, 4900]
        search_start_page = 1
        page_range = await enc._MedicalChronologyExtractor__find_page_range(page_indexes_dict, chunk_indexes, search_start_page)
        assert page_range == (None, None, 1)

    @pytest.mark.asyncio
    async def test_get_page_number_with_valid_parameters(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        reference_text = "exact reference text"
        list_of_page_contents = [
            Document(page_content="text of first page. ", metadata={'page': 1}),
            Document(page_content="text of second page with exact reference text. ", metadata={'page': 2}),
            Document(page_content="text of third page. ", metadata={'page': 3}),
            Document(page_content="text of fourth page. ", metadata={'page': 4})
        ]
        relevant_chunks = [
            Document(page_content="text of first page. text of second page ",
                     metadata={'source': "sample_file", 'start_page': 1, 'end_page': 2}),
            Document(page_content="with exact reference text. text of ",
                     metadata={'source': "sample_file", 'start_page': 2, 'end_page': 3}),
            Document(page_content="third page. text of",
                     metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]

        page_number, filename = await enc._MedicalChronologyExtractor__get_page_number(reference_text, list_of_page_contents,
                                                                                relevant_chunks)
        assert page_number == 2 and filename == "sample_file"

    @pytest.mark.asyncio
    async def test_get_page_number_with_invalid_relevant_chunks(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        reference_text = "exact reference text"
        list_of_page_contents = [
            Document(page_content="text of first page. ", metadata={'page': 1}),
            Document(page_content="text of second page with exact reference text. ", metadata={'page': 2}),
            Document(page_content="text of third page. ", metadata={'page': 3}),
            Document(page_content="text of fourth page. ", metadata={'page': 4})
        ]
        relevant_chunks = [
            Document(page_content="text of first", metadata={'source': "sample_file", 'start_page': 1, 'end_page': 1}),
            Document(page_content="third page. text of",
                     metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]

        page_number, filename = await enc._MedicalChronologyExtractor__get_page_number(reference_text, list_of_page_contents,
                                                                                relevant_chunks)

        assert page_number != 2 and filename == "sample_file"

    @pytest.mark.asyncio
    async def test_get_page_number_with_invalid_page_contents(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        reference_text = "exact reference text"
        list_of_page_contents = [
            Document(page_content="text of first page. ", metadata={'page': 0}),
            Document(page_content="text of second page with exact reference text. ", metadata={'page': 0}),
            Document(page_content="text of third page. ", metadata={'page': 0}),
            Document(page_content="text of fourth page. ", metadata={'page': 0})
        ]
        relevant_chunks = [
            Document(page_content="text of first page. text of second page ",
                     metadata={'source': "sample_file", 'start_page': 1, 'end_page': 2}),
            Document(page_content="with exact reference text. text of ",
                     metadata={'source': "sample_file", 'start_page': 2, 'end_page': 3}),
            Document(page_content="third page. text of",
                     metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]

        page_number, filename = await enc._MedicalChronologyExtractor__get_page_number(reference_text, list_of_page_contents,
                                                                                relevant_chunks)

        assert page_number != 2 and filename == "sample_file"

    @pytest.mark.asyncio
    async def test_get_page_number_with_reference_text_from_different_pages(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        reference_text = "third page. text of fourth page. "
        list_of_page_contents = [
            Document(page_content="text of first page. ", metadata={'page': 1}),
            Document(page_content="text of second page with exact reference text. ", metadata={'page': 2}),
            Document(page_content="text of third page. ", metadata={'page': 3}),
            Document(page_content="text of fourth page. ", metadata={'page': 4})
        ]
        relevant_chunks = [
            Document(page_content="text of first page. text of second page ",
                     metadata={'source': "sample_file", 'start_page': 1, 'end_page': 2}),
            Document(page_content="with exact reference text. text of ",
                     metadata={'source': "sample_file", 'start_page': 2, 'end_page': 3}),
            Document(page_content="third page. text of",
                     metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]

        page_number, filename = await enc._MedicalChronologyExtractor__get_page_number(reference_text, list_of_page_contents,
                                                                                relevant_chunks)
        assert page_number == 3 and filename == "sample_file"
        
    def test_format_date_with_valid_alpha_numeric_input_date(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        is_alpha = True
        input_date = "31 May, 2024"
        formatted_date = enc._MedicalChronologyExtractor__format_date(is_alpha, input_date)
        assert formatted_date == '05-31-2024'

    def test_format_date_with_valid_numeric_input_date(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        is_alpha = False
        input_date = "05-31-2024"
        formatted_date = enc._MedicalChronologyExtractor__format_date(is_alpha, input_date)
        assert formatted_date == '05-31-2024'


    def test_format_date_with_invalid_input_date(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        is_alpha = True
        input_date = 'this is not a date'
        formatted_date = enc._MedicalChronologyExtractor__format_date(is_alpha, input_date)
        assert formatted_date == 'this is not a date'

    @pytest.mark.asyncio
    async def test_post_processing_with_valid_parameters(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        list_of_page_contents = [
            Document(page_content="text of first page. Date-1 : 02/27/2018, Doctor: Mr. xyz, Institution: inst-1, Event-1 with some event ", metadata={'page': 1}),
            Document(page_content="text of second page with exact reference text. Date-2 : 04/2016, Doctor: Mr. abc, Institution: inst-2, Event-2 with some event ", metadata={'page': 2}),
            Document(page_content="text of third page. Date-3 : 08/06/2016, Doctor: Mrs. def, Institution: inst-3, Event-3 with some event ", metadata={'page': 3}),
        ]
        response = """
        here, is the list of encounters with events. 
        [
        ("02/27/2018", "Event-1", "Mr. xyz", "inst-1", "Event-1 with some event"),
         ("04/2016", "Event-2", "Mr. abc", "inst-2", "Event-2 with some event"),
         ("08/06/2016", "Event-3", "Mrs. def", "inst-3", "Event-3 with some event")
         ]
        tell me, if any other assistance is required.
        """
        relevant_chunks = [
            Document(page_content="text of first page. Date-1 : 02/27/2018, Doctor: Mr. xyz, Institution: inst-1, Event-1 with some event text of second page ", metadata={'source': "sample_file", 'start_page': 1, 'end_page': 2}),
            Document(page_content="with exact reference text. Date-2 : 04/2016, Doctor: Mr. abc, Institution: inst-2, Event-2 with some event text of ", metadata={'source': "sample_file", 'start_page': 2, 'end_page': 3}),
            Document(page_content="third page. Date-3 : 08/06/2016, Doctor: Mrs. def, Institution: inst-3, Event-3 with some event text of", metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]
        encounters = await enc._MedicalChronologyExtractor__post_processing(list_of_page_contents, response, relevant_chunks)
        assert encounters == [{"date": "02-27-2018", "doctor_name": "Mr. xyz", "document_name": "sample_file", "event": "Event-1", "hospital_name": "inst-1", "page_no": 1},
                              {"date": "04-01-2016", "doctor_name": "Mr. abc", "document_name": "sample_file", "event": "Event-2", "hospital_name": "inst-2", "page_no": 2},
                              {"date": "08-06-2016", "doctor_name": "Mrs. def", "document_name": "sample_file", "event": "Event-3", "hospital_name": "inst-3", "page_no": 3}]

    @pytest.mark.asyncio
    async def test_post_processing_with_empty_and_invalid_response(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        list_of_page_contents = [
            Document(page_content="text of first page. Date-1 : 02/27/2018, Event-1 with some event ", metadata={'page': 1}),
            Document(page_content="text of second page with exact reference text. Date-2 : 04/2016, Event-2 with some event ", metadata={'page': 2}),
            Document(page_content="text of third page. Date-3 : 08/06/2016, Event-3 with some event ", metadata={'page': 3}),
            Document(page_content="text of fourth page. Date-4 : 05/11/2017, Event-4 with some event ", metadata={'page': 4})
        ]
        empty_response = ""
        invalid_response = """
        here, is the list of encounters with events. 
        (02/27/2018, "Event-1", "Event-1 with some event"),
         (04/2016, "Event-2", "Reference" : "Event-2 with some event"),
         (08/06/2016, "Event-3", {"Reference" : "Event-3 with some event"})]
        tell me, if any other assistance is required.
        """
        relevant_chunks = [
            Document(page_content="text of first page. Date-1 : 02/27/2018, Event-1 with some event text of second page ", metadata={'source': "sample_file", 'start_page': 1, 'end_page': 2}),
            Document(page_content="with exact reference text. Date-2 : 04/2016, Event-2 with some event text of ", metadata={'source': "sample_file", 'start_page': 2, 'end_page': 3}),
            Document(page_content="third page. Date-3 : 08/06/2016, Event-3 with some event text of", metadata={'source': "sample_file", 'start_page': 3, 'end_page': 4})
        ]
        empty_encounters = await enc._MedicalChronologyExtractor__post_processing(list_of_page_contents, empty_response, relevant_chunks)
        invalid_encounters = await enc._MedicalChronologyExtractor__post_processing(list_of_page_contents, invalid_response, relevant_chunks)
        assert empty_encounters == []
        assert invalid_encounters == []

    @pytest.mark.asyncio
    async def test_get_extracted_entities_with_pagewise_text(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        with open('static/Fugarino Dictation_ 06-27-2023_text.json') as f: text = json.load(f)
        assert await enc.get_medical_chronology(text, logger) is not None

    @pytest.mark.asyncio
    async def test_get_extracted_entities_with_empty_pagewise_text(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        text = {}
        template_data = {'medical_chronology': []}
        assert await enc.get_medical_chronology(text, logger) == template_data

    @pytest.mark.asyncio
    async def test_get_extracted_entities_without_pagewise_text(self):
        logger = logging.getLogger()
        enc = MedicalChronologyExtractor(logger)
        text = "This is a extracted text for medical chronology"
        try:
            await enc.get_medical_chronology(text, logger)
        except AttributeError:
            assert True
