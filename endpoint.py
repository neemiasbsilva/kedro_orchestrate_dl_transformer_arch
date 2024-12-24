from fastapi import FastAPI, HTTPException, APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import numpy as np
import tensorflow as tf
import mlflow.tensorflow
from tokenizers import BertWordPieceTokenizer
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

from src.utils import preprocess_prediction_data, create_tf_prediction_dataset

prediction_counter = Counter("model_predictions_total", "Total number of predictions")

with open("endpoint_conf.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

experiment_id = config["endpoint"]["experiment_id"]
metric = config["endpoint"]["metric"]
ascending = config["endpoint"]["ascending"]

params = config["params"]


def load_best_model(experiment_id: str, metric: str, ascending: bool) -> tf.keras.Model:
    """
    Load the best model from a specified MLflow experiment.

    Args:
        experiment_id (str): The experiment ID.
        metric (str): The metric to use for selecting the best model (default is 'accuracy').
        ascending (bool): If True, choose the smallest value of the metric; otherwise, choose the largest (default is False).

    Returns:
        Model: The loaded MLflow model.
    """
    client = mlflow.tracking.MlflowClient()
    best_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if not best_run:
        raise Exception(f"No runs found for experiment ID {experiment_id}")
    
    best_run_id = best_run[0].info.run_id
    model_uri = f"mlruns/{experiment_id}/{best_run_id}/artifacts/distilbert_model"

    return mlflow.tensorflow.load_model(model_uri)


dloaded_model = load_best_model(experiment_id, metric, ascending)

app = FastAPI(
    title="Model Serving to Deploy a Deep Learning Transformer Architecture for the Toxic Comment Classification Problem",
    openapi_prefix="/api/v1/openapi.json"
)

Instrumentator().instrument(app).expose(app)

root_router = APIRouter()

@root_router.get('/')
def index(request: Request):
    """Serve the main HTML page for the API.

    Args:
        request (Request): Contains request-specific information.

    Raises:
        HTTPException: If an error occurs while processing the request.

    Returns:
        HTMLResponse: The HTML content for the homepage.
    """
    body = """"
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                color: #333;
                padding: 20px;
                margin: 0;
            }
            h1 {
                color: #4CAF50;
                text-align: center;
            }
            .container {
                max-width: 600px;
                margin: auto;
                padding: 20px;
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], input[type="file"] {
                width: 100%;
                padding: 10px;
                margin: 5px 0 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Toxic Comment Classification API</h1>
            <form id="uploadForm" action="/predict" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="message">Enter a message:</label>
                    <input type="text" id="message" name="message" placeholder="Enter your text here">
                </div>
                <div class="form-group">
                    <label for="file">Upload a CSV file:</label>
                    <input type="file" id="file" name="file">
                </div>
                <button type="submit" class="button">Submit</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=body)
    

@root_router.post("/predict", status_code=200)
def predict(message: str = Form(...), file: UploadFile = File(None)):
    """
    Predict the output for the given input message or process the uploaded CSV file.

    Args:
        message (str): Input text message.
        file (UploadFile, optional): Uploaded CSV file containing messages.

    Returns:
        dict: Predictions from the model.
    """
    try:
        if file:
            import pandas as pd
            df = pd.read_csv(file.file)
            if df.shape[1] == 0:
                raise HTTPException(status_code=400, detail="Uploaded file has no columns")
            
            inputs = df.content.tolist() 

            if not inputs:
                raise HTTPException(status_code=400, detail="No valid data in the uploaded file.")
        else:
            if not message.strip():
                raise HTTPException(status_code=400, detail="Message input is empty.")
            inputs = [message.strip()]

        fast_tokenizer = BertWordPieceTokenizer(params["fast_tokenizer_path"] + "/vocab.txt")
        inputs_array = np.array(inputs)
        X_pred = preprocess_prediction_data(inputs_array, fast_tokenizer, maxlen=256)
        prediction_dataset = create_tf_prediction_dataset(X_pred, params)

        predictions = dloaded_model.predict(prediction_dataset)
        prediction_counter.inc()
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@root_router.get("/predict", status_code=200)
def predict_get():
    """
    Serve a simple HTML form or a message for the predict endpoint.
    """
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Predict Endpoint</title>
    </head>
    <body>
        <h1>Predict Endpoint</h1>
        <p>This endpoint only supports POST requests. Please use a tool like Postman or a JavaScript-based request from the main page to send predictions.</p>
        <p>If you want to test it manually, use the following JSON payload in a POST request:</p>
        <pre>
        {
            "inputs": ["Example input text"]
        }
        </pre>
    </body>
    </html>
    """)
    
app.include_router(root_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3400",
        "http://localhost:8400",
        "https://localhost:3400",
        "https://localhost:8400"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")