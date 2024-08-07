import os


async def get_project_id_and_document(document_path):
    document_name = os.path.basename(document_path)
    project_id = document_path.split('/')[2]
    return project_id, document_name
