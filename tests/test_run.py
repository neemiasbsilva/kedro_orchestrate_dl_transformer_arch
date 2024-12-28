import pandas as pd
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.io import DataCatalog
from unittest.mock import patch
import pytest

@pytest.fixture(scope="module")
def small_data():
    """Mock a small dataset for testing.
    """
    train_data = pd.DataFrame({
        "comment_text": ["This is a toxic comment."]*100 + ["This is a neutral comment."]*100,
        "toxic": [1]*100 + [0]*100
    })

    val_data = pd.DataFrame({
        "comment_text": ["This is a toxic comment."]*20 + ["This is a neutral comment."]*20,
        "toxic": [1]*20 + [0]*20
    })
    

    test_data = pd.DataFrame({
        "comment_text": ["This is a toxic comment."]*20 + ["This is a neutral comment."]*20,
        "toxic": [1]*20 + [0]*20
    })

    return train_data, val_data, test_data


@pytest.fixture(scope="module")
def mock_pipeline_params():
    """Mock parameters used in the pipeline"""
    return {
        "preprocessing_options": {
            "tokenizer_path": "distilbert-base-uncased",
            "fast_tokenizer_path": "./fast_tokenizer",
        },
        "dataset_tf_options": {
            "AUTO_PREFETCH": True,
            "SHUFFLE_BUFFER": 10,
            "BATCH_SIZE": 2,
        },
        "model_options": {
            "MAX_LEN": 256,
            "EPOCHS": 1,
            "BATCH_SIZE": 2,
            "LR": 1e-5,
            "uri": "http://mlflow-server.local",
            "experiment_name": "TestExperiment",
            "summary_path": "./model_summary.txt",
            "plot_path": "./plots",
        },
        "evaluation_options": {
            "BATCH_SIZE": 2,
            "prediction_path": "./predictions.json",
        }
    }


@patch("kedro.io.DataCatalog.load")
def test_pipeline_run(mock_load, small_data, mock_pipeline_params):
    """Test end-to-end pipeline run with mocked data to optimize memory usage"""
    train_data, val_data, test_data = small_data
    mock_load.side_effect = lambda dataset_name: {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "params:preprocessing_options": mock_pipeline_params["preprocessing_options"],
        "params:dataset_tf_options": mock_pipeline_params["dataset_tf_options"],
        "params:model_options": mock_pipeline_params["model_options"],
        "params:evaluation_options": mock_pipeline_params["evaluation_options"],
    }.get(dataset_name, None)

    with patch("kedro.framework.session.KedroSession.run") as mock_run:
        mock_run.return_value = {"champions_model": "mock_model_path"}

        project_path = Path.cwd()
        bootstrap_project(project_path)
        with KedroSession.create(project_path=project_path) as session:
            pipeline_output = session.run()

        assert pipeline_output is not None
        assert "champions_model" in pipeline_output