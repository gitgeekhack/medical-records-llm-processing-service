import json
from json import JSONDecodeError
import pytest
import logging

from langchain.chains.qa_with_sources.stuff_prompt import template

from app.constant import MedicalInsights
from app.service.nlp_extractor.history_extractor import HistoryExtractor

pytest_plugins = ('pytest_asyncio',)

class TestHistoryExtractor:
    @pytest.mark.asyncio
    async def test_post_processing_with_curly_braces_json(self):
        logger = logging.getLogger()
        he = HistoryExtractor(logger)
        text = """
        {"text": "This is  a text with curly braces"}
        """
        assert isinstance(await he._HistoryExtractor__post_processing(text), dict)

    @pytest.mark.asyncio
    async def test_post_processing_without_curly_braces(self):
        logger = logging.getLogger()
        he = HistoryExtractor(logger)
        text = """
        text: This is  a text without curly braces
        """
        try:
            assert await he._HistoryExtractor__post_processing(text)
        except JSONDecodeError:
            assert True

    @pytest.mark.asyncio
    async def test_get_history_with_pagewise_text(self):
        logger = logging.getLogger()
        he = HistoryExtractor(logger)
        with open('static/Fugarino Dictation_ 06-27-2023_text.json') as f: text = json.load(f)
        assert await he.get_history(text) is not None

    @pytest.mark.asyncio
    async def test_get_history_with_empty_pagewise_text(self):
        logger = logging.getLogger()
        he = HistoryExtractor(logger)
        text = {}
        template_data = MedicalInsights.TemplateResponse.HISTORY_TEMPLATE_RESPONSE
        assert await he.get_history(text) == {"general_history": template_data}

    @pytest.mark.asyncio
    async def test_get_patient_demographics_without_pagewsie_text(self):
        logger = logging.getLogger()
        he = HistoryExtractor(logger)
        text = "Patient has a history of drinking alcohol and smoking tobacco"
        try:
            await he.get_history(text)
        except AttributeError:
            assert True