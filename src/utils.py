import logging
import numpy as np
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf

def preprocess_prediction_data(
        texts: list, 
        fast_tokenizer: BertWordPieceTokenizer,
        maxlen: int=256
) -> np.array:
    """
    Preprocesses the prediction dataset by tokenizing the input texts.

    Args:
        texts (list): A list of text samples to preprocess.
        fast_tokenizer (BertWordPieceTokenizer): Pretrained tokenizer.
        maxlen (int, optional): Maximum token length for padding/truncation. Defaults to 512.

    Returns:
        np.array: Tokenized and padded input features.
    """
    logging.info("Starting preprocessing for prediction dataset.")
    fast_tokenizer.enable_truncation(max_length=maxlen)
    fast_tokenizer.enable_padding(length=maxlen)

    all_ids = []

    for text in texts:
        encoded = fast_tokenizer.encode(text)
        all_ids.append(encoded.ids)

    logging.info("Prediction dataset tokenization completed")
    return np.array(all_ids)


def create_tf_prediction_dataset(X_pred: np.array, params: dict) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset for prediction data.

    Args:
        X_pred (numpy.ndarray): Features for the prediction dataset.
        params (dict): Dictionary of parameters for dataset creation.

    Returns:
        tf.data.Dataset: TensorFlow dataset for predictions.
    """
    logging.info("Creating TensorFlow dataset for predictions.")
    try:
        AUTO = tf.data.AUTOTUNE if params.get("AUTO_PREFETCH", True) else tf.data.Dataset.prefetch
        prediction_dataset = tf.data.Dataset.from_tensor_slices(X_pred).batch(params["BATCH_SIZE"]).prefetch(AUTO)
        logging.info("Prediction dataset created successfully.")
        return prediction_dataset
    except Exception as e:
        logging.error(f"Error in creating prediction TensorFlow dataset: {e}")
        raise