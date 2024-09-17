FROM python:3.9-slim

WORKDIR /llm-processing-service

COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip==24.2
RUN pip install -r requirements.txt
COPY ./app ./app
COPY run.py ./run.py

CMD ["python", "run.py"]
