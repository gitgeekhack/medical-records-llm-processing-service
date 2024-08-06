FROM python:3.9-slim

WORKDIR /start-textract-async

COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip==24.2
RUN pip install -r requirements.txt
COPY ./app ./app
COPY textract_run.py ./textract_run.py

CMD ["python", "textract_run.py"]
