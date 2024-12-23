from fastapi import FastAPI, HTTPException
import uvicorn
import yaml
import numpy as np
import tensorflow as tf
import mlflow.tensorflow
from tokenizers import BertWordPieceTokenizer

from src.utils import preprocess_prediction_data, create_tf_prediction_dataset

app = FastAPI()

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

@app.post("/predict")
def predict(data: dict):
    """
    Predict the output for the given input data using the loaded model.

    Args:
        data (dict): Input data in JSON format with a key 'inputs'.

    Returns:
        dict: Predictions from the model.
    """
    inputs = data.get("inputs")

    fast_tokenizer = BertWordPieceTokenizer(params["fast_tokenizer_path"] + "/vocab.txt")
    if inputs is None:
        raise HTTPException(status_code=400, detail="Input data missing")
    
    try:
        inputs_array = np.array(inputs)
        X_pred = preprocess_prediction_data(inputs_array, fast_tokenizer, maxlen=256)
        prediction_dataset = create_tf_prediction_dataset(X_pred, params)

        predictions = dloaded_model.predict(prediction_dataset)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)