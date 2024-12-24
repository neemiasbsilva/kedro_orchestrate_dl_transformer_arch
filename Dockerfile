FROM python:3.9

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY endpoint.py endpoint_conf.yml /app/

COPY data/05_model_input /app/data/05_model_input
COPY src/toxic_comment_classification_kedro/pipelines/data_science/nodes.py /app/toxic_comment_classification_kedro/pipelines/data_science/nodes.py
COPY src/utils.py /app/src/utils.py
COPY mlruns /app/mlruns

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "endpoint:app", "--host", "0.0.0.0", "--port", "8001"]
