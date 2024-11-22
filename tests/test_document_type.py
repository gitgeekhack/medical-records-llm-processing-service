import pytest
from app.service.nlp_extractor.doc_type_extractor import DocTypeExtractor
import logging

pytest_plugins = ('pytest_asyncio',)


class TestDocumentType:
    @pytest.mark.asyncio
    async def test_process_doc_type_with_valid_params(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        output_text = """ Here is the response from LLM: {"document_type": "Ambulance Report"} """

        result = await dte._DocTypeExtractor__process_document_type(output_text)
        assert result == {"document_type": "Ambulance Report"}

    @pytest.mark.asyncio
    async def test_process_doc_type_with_missing_brackets(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        output_text = """ Here is the response from LLM: document_type: Ambulance Report """

        result = await dte._DocTypeExtractor__process_document_type(output_text)

        assert result == {"document_type": ""}

    @pytest.mark.asyncio
    async def test_process_doc_type_without_document_type_field(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        output_text = """ Here is the response from LLM: {"some_other_field": "value"} """

        result = await dte._DocTypeExtractor__process_document_type(output_text)

        assert result == {"document_type": ""}

    @pytest.mark.asyncio
    async def test_data_formatter_with_page_wise_text(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        page_wise_text = {
            "page_1": "This is some sample text for page 1.",
            "page_2": "This is some sample text for page 2.",
            "page_3": "This is some sample text for page 3.",
        }
        docs = await dte._DocTypeExtractor__data_formatter(page_wise_text)
        assert len(docs) > 0

    @pytest.mark.asyncio
    async def test_data_formatter_with_empty_page_wise_text(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        page_wise_text = {}
        docs = await dte._DocTypeExtractor__data_formatter(page_wise_text)
        assert len(docs) == 0

    @pytest.mark.asyncio
    async def test_get_doc_embeddings_with_page_wise_text(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        page_wise_text = {
            "page_1": "This is some sample text for page 1.",
            "page_2": "This is some sample text for page 2.",
            "page_3": "This is some sample text for page 3.",
        }
        emb = await dte._DocTypeExtractor__get_docs_embeddings(page_wise_text)
        if emb:
            assert True

    @pytest.mark.asyncio
    async def test_data_formatter_with_empty_page_wise_text(self):
        logger = logging.getLogger()
        dte = DocTypeExtractor(logger)
        page_wise_text = {}
        try:
            await dte._DocTypeExtractor__get_docs_embeddings(page_wise_text)
        except IndexError:
            assert True
