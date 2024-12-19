import logging
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import TFDistilBertModel
import mlflow
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DistilBertLayer(Layer):
    def __init__(self, **kwargs):
        super(DistilBertLayer, self).__init__(**kwargs)
        logging.info("Initializing DistilBertLayer.")
        self.transformer_layer = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

    def call(self, inputs):
        logging.info("Forward pass through DistilBERT")
        # Forward pass through DistilBERT
        output = self.transformer_layer(input_ids=inputs)[0]  # Sequence output
        logging.debug(f"Output shape after DistilBERT layer: {output.shape}")
        return output

def build_model(params: dict) -> Model:
    """Function for get the BERT training model

    Args:
        transformer (TFDistilBertModel): Pretrained model
        max_len (int, optional): Number of dimensions of the last layer of pretrained model. Defaults to 512.

    Returns:
        Model: Keras Model compiled
    """
    logging.info(f'uilding model with max_len={params["MAX_LEN"]}')
    input_word_ids = Input(
        shape=(params["MAX_LEN"],),
        dtype=tf.int32,
        name="input_word_ids"
    )

    sequence_output = DistilBertLayer()(input_word_ids)

    cls_token = sequence_output[:, 0, :]
    logging.debug(f"CLS token shape: {cls_token.shape}")

    out = Dense(1, activation="sigmoid")(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    logging.info("Compiling model.")
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    logging.info("Model built and compiled successfully.")
    return model


def save_training_history_with_mlflow(history: dict, params: dict, metrics=("accuracy", "loss"), log_to_mlflow=True) -> None:
    """
    Plots training and validation metrics for a given training history and logs to MLflow.

    Args:
        history (dict): Dictionary containing training history (e.g., history.history).
        metrics (tuple): Tuple of metrics to plot (default: ("accuracy", "loss")).
        log_to_mlflow (bool): If True, logs the plots to MLflow.
    """
    

    # Log plot to MLflow
    if log_to_mlflow:
        for metric in metrics:
            plot_path = os.path.join(params["plot_path"], f"{metric}_plot.png")
            plt.figure(figsize=(6, 4))

            train_metric = history.get(metric)
            val_metric = history.get(f"val_{metric}")

            if train_metric:
                plt.plot(train_metric, label=f'Train {metric.capitalize()}')
            if val_metric:
                plt.plot(val_metric, label=f'Validation {metric.capitalize()}')

            plt.title(f'{metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()


def train_model_with_mlflow(dl_model: tf.keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, X_train: np.array, params: dict) -> None:
    """
    Trains a deep learning model, logs the training process, and saves artifacts to MLflow.

    Args:
        dl_model (tf.keras.Model): The deep learning model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        X_train (np.array): Training data used to calculate steps per epoch.
        params (dict): Parameters for training and MLflow logging (e.g., batch size, epochs, learning rate, URI).

    Returns:
        None
    """
    train_dataset = train_dataset[0]
    val_dataset = val_dataset[0]

    mlflow.set_tracking_uri(params["uri"])
    experiment = mlflow.set_experiment(params["experiment_name"])

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param("batch_size", params["BATCH_SIZE"])
        mlflow.log_param("epochs", params["EPOCHS"])
        mlflow.log_param("learning_rate", params["LR"])

        mlflow.tensorflow.autolog()

        n_steps = X_train.shape[0] // params["BATCH_SIZE"]

        train_history = dl_model.fit(
            train_dataset,
            steps_per_epoch=n_steps,
            validation_data=val_dataset,
            epochs=params["EPOCHS"]

        )


        save_training_history_with_mlflow(train_history.history, params)

        for epoch, accuracy in enumerate(train_history.history["accuracy"]):
            mlflow.log_metric(f"train_accuracy_epoch_{epoch}", accuracy)

        for epoch, accuracy in enumerate(train_history.history["val_accuracy"]):
            mlflow.log_metric(f"val_accuracy_epoch_{epoch}", accuracy)

        mlflow.tensorflow.log_model(dl_model, artifact_path="distilbert_model", registered_model_name="DistilBERT_Toxic_Comment_Classification")
        with open(params["summary_path"], "w") as f:
            dl_model.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(params["summary_path"])

        logging.info(f"RUN ID: {mlflow.active_run().info.run_id}")

    return dl_model, experiment.experiment_id


def evaluation_model_step(dl_model: tf.keras, experiment_id: str, test_dataset: tf.data.Dataset, params: dict) -> None:
    test_dataset = test_dataset[0]
    mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run(experiment_id=experiment_id):
        evaluation_results = dl_model.evaluate(test_dataset, batch_size=params.get("BATCH_SIZE", 32), return_dict=True)

        for metric, value in evaluation_results.items():
            mlflow.log_metric(metric, value)

        # Save model predictions (optional)
        predictions = dl_model.predict(test_dataset)
        predictions_path = params["prediction_path"]
        with open(predictions_path, "w") as json_file:
            json.dump(predictions.tolist(), json_file)
        mlflow.log_artifact(predictions_path)

        # Fetch and compare with the previous best model
        client = mlflow.tracking.MlflowClient()
        best_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )

        if best_run:
            best_run = best_run[0]
            logging.info(f"Comparing with the best model from RUN ID: {best_run.info.run_id}")

            previous_accuracy = best_run.data.metrics.get("accuracy", 0)
            current_accuracy = evaluation_results.get("accuracy", 0)

            if current_accuracy > previous_accuracy:
                logging.info("The current model outperforms the previous best model.")
                champions_model = dl_model
            else:
                logging.info("The previous model performed better.")
                champions_model = best_run
        else:
            logging.info("No previous runs found for comparison.")
            champions_model = dl_model
        
        logging.info(f"Evaluation completed and logged in experiment ID: {experiment_id}")
        return champions_model
