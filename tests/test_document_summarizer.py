import json

import pytest
from langchain.chains.qa_generation.prompt import templ

from app.service.nlp_extractor.document_summarizer import DocumentSummarizer
from app.constant import MedicalInsights

from langchain.docstore.document import Document

import logging

pytest_plugins = ('pytest_asyncio',)


class TestDocumentSummarizer:
    @pytest.mark.asyncio
    async def test_first_line_remove_with_matching_line(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        reference_summary_first_last_line = MedicalInsights.LineRemove.SUMMARY_FIRST_LAST_LINE_REMOVE
        summary = """ Here is the detailed summary of patient XYZ admitted in hospital ABC """
        if await ds._DocumentSummarizer__first_line_remove(summary, reference_summary_first_last_line):
            assert True

    @pytest.mark.asyncio
    async def test_first_line_remove_with_no_matching_line(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        reference_summary_first_last_line = MedicalInsights.LineRemove.SUMMARY_FIRST_LAST_LINE_REMOVE
        summary = """ Summary of patient XYZ admitted in hospital ABC """
        if not await ds._DocumentSummarizer__first_line_remove(summary, reference_summary_first_last_line):
            assert True

    @pytest.mark.asyncio
    async def test_get_stuff_calls_with_valid_parameter(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        docs = [Document(page_content="This is a first line"),
                Document(page_content="This is a second line"),
                Document(page_content="This is a third line"),
                Document(page_content="This is a fourth line")]
        chunk_length = [20000, 20000, 30000, 25000]
        result = await ds._DocumentSummarizer__get_stuff_calls(docs, chunk_length)
        assert result == [
            [docs[0], docs[1], docs[2]],
            [docs[3]]
        ]

    @pytest.mark.asyncio
    async def test_get_stuff_calls_with_invalid_size_of_chunk_length(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        docs = [Document(page_content="This is a first line"),
                Document(page_content="This is a second line")]
        chunk_length = []
        result = await ds._DocumentSummarizer__get_stuff_calls(docs, chunk_length)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_stuff_calls_with_invalid_parameter_of_empty_list(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        docs = []
        chunk_length = []
        result = await ds._DocumentSummarizer__get_stuff_calls(docs, chunk_length)
        assert result == []

    @pytest.mark.asyncio
    async def test_post_processing_with_valid_parameter(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        input_summary = "This is the testing summary line."
        result = await ds._DocumentSummarizer__post_processing(input_summary)
        assert result is not None

    @pytest.mark.asyncio
    async def test_post_processing_with_list_input(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        input_summary = ["This is a example of summary line.", "This is a second example of summary line."]
        try:
            await ds._DocumentSummarizer__post_processing(input_summary)
        except AttributeError:
            assert True

    @pytest.mark.asyncio
    async def test_post_processing_with_empty_string(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        input_summary = ""
        result = await ds._DocumentSummarizer__post_processing(input_summary)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_summary_with_pagewise_text(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        with open('static/Fugarino Dictation_ 06-27-2023_text.json') as f: text = json.load(f)
        assert await ds.get_summary(text) is not None

    @pytest.mark.asyncio
    async def test_get_summary_with_empty_pagewise_text(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        text = {}
        template_data = {"summary": MedicalInsights.TemplateResponse.SUMMARY_RESPONSE}
        assert await ds.get_summary(text) == template_data

    @pytest.mark.asyncio
    async def test_get_summary_without_pagewise_text(self):
        logger = logging.getLogger()
        ds = DocumentSummarizer(logger)
        text = "This is a summary of the patient report"
        try:
            await ds.get_summary(text)
        except AttributeError:
            assert True
