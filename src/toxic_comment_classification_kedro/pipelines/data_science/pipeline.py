from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_model, train_model_with_mlflow, evaluation_model_step#evaluate_model, split_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_model,
                inputs=["params:model_options"],
                outputs="dl_model",
                name="build_dl_model_step"
            ),
            node(
                func=train_model_with_mlflow,
                inputs=["dl_model", "train_dataset", "val_dataset", "X_train", "params:model_options"],
                outputs=["dl_model_trained", "experiment_id"],
                name="train_dl_model_step"
            ),
            node(
                func=evaluation_model_step,
                inputs=["dl_model_trained", "experiment_id", "test_dataset", "params:evaluation_options"],
                outputs="champions_model"
            )
        ]
    )
