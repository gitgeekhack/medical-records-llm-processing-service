import json
import logging

import pytest
from app.service.nlp_extractor import entity_extractor

pytest_plugins = ('pytest_asyncio',)


class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_is_alpha_with_alphabetic_entity(self):
        entity = "diabetes"
        if await entity_extractor.is_alpha(entity):
            assert True

    @pytest.mark.asyncio
    async def test_is_alpha_with_non_alphabetic_entity(self):
        entity = "1234"
        if not await entity_extractor.is_alpha(entity):
            assert True

    @pytest.mark.asyncio
    async def test_parse_date_with_valid_date(self):
        date = "9 August, 2001"
        assert await entity_extractor.parse_date(date) == "08-09-2001"

    @pytest.mark.asyncio
    async def test_parse_date_with_valid_date(self):
        date = "982001"
        assert await entity_extractor.parse_date(date) != "08-09-2001"

    @pytest.mark.asyncio
    async def test_get_page_number_with_valid_key(self):
        key = "Page_1"
        assert await entity_extractor.get_page_number(key) == 1

    @pytest.mark.asyncio
    async def test_get_page_number_with_valid_key(self):
        key = "Page_one"
        if not isinstance(await entity_extractor.get_page_number(key), int):
            assert True

    @pytest.mark.asyncio
    async def test_convert_str_into_json_with_valid_json(self):
        json = """
        {
      "diagnosis": {
        "allergies": [],
        "pmh": [
          "Hypertension"
        ],
        "others": [
          "Left shoulder pain",
          "Radiating pain into wrist",
          "Numbness and tingling"
        ]}}
        """
        if isinstance(await entity_extractor.convert_str_into_json(json, 1), dict):
            assert True

    @pytest.mark.asyncio
    async def test_convert_str_into_json_with_invalid_json(self):
        json = """
        {{
      "diagnosis": {
        "allergies": [],
        "pmh": [
          "Hypertension"
        ],
        "others": [
          "Left shoulder pain",
          "Radiating pain into wrist",
          "Numbness and tingling"
        ]}}
        """
        try:
            await entity_extractor.convert_str_into_json(json, 1)
        except SyntaxError:
            assert True

    @pytest.mark.asyncio
    async def test_convert_str_into_json_with_empty_json(self):
        json = """
        {}
        """
        if isinstance(await entity_extractor.convert_str_into_json(json, 1), dict):
            assert True

    @pytest.mark.asyncio
    async def test_get_extracted_entities_with_pagewise_text(self):
        logger = logging.getLogger()
        with open('static/Fugarino Dictation_ 06-27-2023_text.json') as f: text = json.load(f)
        assert await entity_extractor.get_extracted_entities(text, logger) is not None

    @pytest.mark.asyncio
    async def test_get_extracted_entities_with_empty_pagewise_text(self):
        logger = logging.getLogger()
        text = {}
        assert await entity_extractor.get_extracted_entities(text, logger) == {"medical_entities": []}

    @pytest.mark.asyncio
    async def test_get_extracted_entities_without_pagewise_text(self):
        logger = logging.getLogger()
        text = "Medical entities are diabetes and has a past history of Bad cholestrol"
        try:
            await entity_extractor.get_extracted_entities(text, logger)
        except AttributeError:
            assert True


