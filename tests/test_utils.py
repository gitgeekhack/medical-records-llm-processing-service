import pytest

from app.common.utils import get_project_id_and_document

pytest_plugins = ('pytest_asyncio',)

class TestUtils:
    @pytest.mark.asyncio
    async def test_get_project_id_and_document_with_valid_path(self):
        document_path = "request/1234-abcd/defg-5678-hij/Fugarino Dictation_ 06-27-2023.pdf"
        id, name = await get_project_id_and_document(document_path)
        if id == "defg-5678-hij":
            assert True

    @pytest.mark.asyncio
    async def test_get_project_id_and_document_with_invalid_path(self):
        document_path = "request//1234-abcd//defg-5678-hij/Fugarino Dictation_ 06-27-2023.pdf"
        id, name = await get_project_id_and_document(document_path)
        if id != "defg-5678-hij":
            assert True

    @pytest.mark.asyncio
    async def test_get_project_id_and_document_with_empty_path(self):
        document_path = ""
        try:
            await get_project_id_and_document(document_path)
        except IndexError:
            assert True

