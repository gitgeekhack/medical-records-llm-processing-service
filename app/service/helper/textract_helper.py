import boto3
import json

from app.constant import AWS

textract_client = boto3.client('textract', region_name=AWS.BotoClient.AWS_DEFAULT_REGION)


# document_text = {}
# # res = client.get_document_text_detection(JobId="1b0fa10fbaf40f45acf154875fa50ace1bf805ea97bbd510ab82cd9062fb6f7e")
#
# with open('res.json', 'r') as f:
#     res = f.read()
# res = eval(res)
#
# for block in res['Blocks']:
#     if block['BlockType'] == 'LINE':
#         document_text['page_' + str(block['Page'])] = document_text.get('page_' + str(block['Page']), '') + block[
#             'Text'] + ' '
#
# print(res)


async def get_page_wise_text(input_message, page_wise_text={}, next_token=None):
    if next_token:
        textract_response = textract_client.get_document_text_detection(JobId=input_message['JobId'],
                                                                        NextToken=next_token)
    else:
        textract_response = textract_client.get_document_text_detection(JobId=input_message['JobId'])

    for block in textract_response['Blocks']:
        if block['BlockType'] == 'LINE':
            page_wise_text['page_' + str(block['Page'])] = (
                    page_wise_text.get('page_' + str(block['Page']), '') + block['Text'] + ' ')

    if 'NextToken' in textract_response:
        await get_page_wise_text(input_message, page_wise_text, next_token=textract_response['NextToken'])

    # result = json.dumps(page_wise_text)
    # result = result.encode("utf-8")
    # await s3_utils.upload_object(AWS.S3.MEDICAL_BUCKET_NAME, s3_textract_path, result, AWS.S3.ENCRYPTION_KEY)

    return page_wise_text
