from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_tf_datasets, preprocessing

# from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocessing,
                inputs=[
                    "train_data",
                    "val_data",
                    "test_data",
                    "params:preprocessing_options",
                ],
                outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
                name="preprocessing_raw_data_step",
            ),
            node(
                func=create_tf_datasets,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "X_test",
                    "y_test",
                    "params:dataset_tf_options",
                ],
                outputs=["train_dataset", "val_dataset", "test_dataset"],
                name="create_tf_datasets_step",
            ),
        ]
    )
