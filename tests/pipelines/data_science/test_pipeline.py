import logging
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
# from kedro.io import DataCatalog
# from kedro.runner import SequentialRunner
from transformers import DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer
from toxic_comment_classification_kedro.pipelines.data_science.nodes import (
    build_model,
    train_model_with_mlflow,
    evaluation_model_step,
)
# from kedro.pipeline import Pipeline, node
# from toxic_comment_classification_kedro.pipelines.data_science.pipeline import create_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def dummy_params():
    logger.info("Setting up dummy parameters")
    return {
        "MAX_LEN": 128,
        "BATCH_SIZE": 32,
        "EPOCHS": 1,
        "LR": 1e-5,
        "uri": "mlruns/test",
        "experiment_name": "test_experiment",
        "plot_path": "data/09_tracking/",
        "prediction_path": "data/09_tracking/test_predictions.json",
        "summary_path": "data/07_model_output/test_model_summary.txt"
    }

@pytest.fixture(scope="module")
def dummy_dataset(dummy_params):
    logger.info("Creating textual dummy datasets for training, validation, and testing.")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_train = ["This is a toxic comment."]*100 + ["This is a neutral comment."]*100
    labels_train = [1] * 100 + [0] * 100
    text_val = ["Validation toxic comment."]*20 + ["Validation neutral comment."]*20
    labels_val = [1] * 20 + [0] * 20
    text_test = ["Test toxic comment."]*20 + ["Test neutral comment."]*20
    labels_test = [1] * 20 + [0] * 20

    def preprocess(texts, labels):
        vocab = tokenizer.get_vocab()
        
        fast_tokenizer = BertWordPieceTokenizer(vocab=vocab)

        all_ids = []

        for i in range(0, len(texts), 256):
            text_chunk = texts[i:i+256]
            logging.debug("Tokenizing chunk %d-%d", i, i + 256)

            encs = fast_tokenizer.encode_batch(text_chunk)
            all_ids.extend([
                enc.ids + [0] * (dummy_params["MAX_LEN"] - len(enc.ids))
                if len(enc.ids) < dummy_params["MAX_LEN"] else enc.ids[:dummy_params["MAX_LEN"]]
                for enc in encs
            ])

        logging.info("Tokenization completed for %d texts.", len(texts))
        return (np.array(all_ids), np.array(labels))
    
    train_data = preprocess(text_train, labels_train)
    val_data = preprocess(text_val, labels_val)
    test_data = preprocess(text_test, labels_test)

    logger.debug(f"train_data: {train_data}")
    logger.debug(f"val_data: {val_data}")
    logger.debug(f"test_data: {test_data}")

    return train_data, val_data, test_data


def test_build_model(dummy_params):
    logger.info("Testing model building function")
    model = build_model(dummy_params)
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0
    logger.info("Model built sucessfully with the correct structure")


@patch("src.toxic_comment_classification_kedro.pipelines.data_science.nodes.train_model_with_mlflow")
def test_train_model_with_mlflow(mock_train_model_with_mlflow, dummy_dataset, dummy_params):
    logger.info("Testing model training with MLflow")
    dummy_data = dummy_dataset
    assert len(dummy_data) == 3, "Fixture did not return the expected number of datasets."
    train_data, val_data, _ = dummy_dataset
    AUTO = tf.data.AUTOTUNE 
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).repeat().shuffle(1000).batch(
        dummy_params["BATCH_SIZE"]).prefetch(AUTO)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data).batch(
        dummy_params["BATCH_SIZE"]).cache().prefetch(AUTO)

    model = build_model(dummy_params)
    logger.debug(f"{dummy_params}")
    trained_model, experiment_id = train_model_with_mlflow(
        model,
        np.array([train_ds]),
        np.array([val_ds]),
        np.array(train_data[0]),
        dummy_params
    )
    assert isinstance(trained_model, tf.keras.Model)
    assert experiment_id is not None

    return trained_model, experiment_id


@pytest.fixture(scope="module")
def trained_model_and_experiment(dummy_dataset, dummy_params):
    train_data, val_data, _ = dummy_dataset
    AUTO = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).repeat().shuffle(1000).batch(
        dummy_params["BATCH_SIZE"]).prefetch(AUTO)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data).batch(
        dummy_params["BATCH_SIZE"]).cache().prefetch(AUTO)

    model = build_model(dummy_params)
    
    trained_model, experiment_id = train_model_with_mlflow(
        model,
        np.array([train_ds]),
        np.array([val_ds]),
        np.array(train_data[0]),
        dummy_params
    )
    return trained_model, experiment_id


@patch("src.toxic_comment_classification_kedro.pipelines.data_science.nodes.evaluation_model_step")
def test_evaluation_model_step(mock_evaluation_model_step, trained_model_and_experiment, dummy_params, dummy_dataset):
    logger.info("Testing model evaluation")
    _, _, test_data = dummy_dataset
    AUTO = tf.data.AUTOTUNE
    test_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(
        dummy_params["BATCH_SIZE"]).cache().prefetch(AUTO)
    
    trained_model, experiment_id = trained_model_and_experiment


    champions_model = evaluation_model_step(
        trained_model,
        experiment_id=experiment_id,
        test_dataset=np.array([test_ds]),
        params=dummy_params
    )

    assert isinstance(champions_model, tf.keras.Model)
    logger.info("Model evaluation completed sucessfully, champions model identified.")
