import logging
import pandas as pd
import numpy as np
from tokenizers import BertWordPieceTokenizer
import transformers
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocessing(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, params: dict) -> tuple:
    """
    Preprocesses the data for model training and evaluation.
    
    Args:
        train_data (pd.DataFrame): Training dataset containing 'comment_text' and 'toxic' columns.
        val_data (pd.DataFrame): Validation dataset containing 'comment_text' and 'toxic' columns.
        test_data (pd.DataFrame): Test dataset containing 'content' column.
        params (dict): Dictionary of parameters for preprocessing.
    
    Returns:
        tuple: Processed training, validation, and test embeddings, along with their corresponding labels.
    """
    logging.info("Starting preprocessing function.")
    train_data = train_data.iloc[:100]
    val_data = val_data.iloc[:50]
    test_data = test_data.iloc[50:100]
    test_data = test_data.reset_index(drop=True)

    logging.info("Loading tokenizer from path: %s", params["tokenizer_path"])
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(params["tokenizer_path"])
    tokenizer.save_pretrained(params["fast_tokenizer_path"])

    logging.info("Loading fast tokenizer vocab.txt from path: %s", params["fast_tokenizer_path"])
    fast_tokenizer = BertWordPieceTokenizer(params["fast_tokenizer_path"]+"/vocab.txt")

    logging.info("Preprocessing training embeddings.")
    X_train = preprocess_embeddings(train_data.comment_text.astype(str), fast_tokenizer, maxlen=256)
    y_train = train_data.toxic.values
    
    logging.info("Preprocessing validation embeddings.")
    X_val = preprocess_embeddings(val_data.comment_text.astype(str), fast_tokenizer, maxlen=256)
    y_val = val_data.toxic.values

    logging.info("Preprocessing test embeddings.")
    X_test = preprocess_embeddings(test_data.comment_text.astype(str), fast_tokenizer, maxlen=256)
    y_test = test_data.toxic.values

    logging.info("Preprocessing completed successfully.")
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_embeddings(texts: list, tokenizer: BertWordPieceTokenizer, chunk_size=256, maxlen=512)-> np.array:
    """

    Args:
        texts (list): all the input samples that will be used for tokenization
        tokenizer (BertWordPieceTokenizer): Distil-BERT tokenizer
        chunk_size (int, optional): The batch_size that will be passed for tokenization. Defaults to 256.
        maxlen (int, optional): Dimensions of the embeddings returned in the tokenization process. Defaults to 512.

    Returns:
        np.array: the array_id returned after preprocess
    """
    logging.info("Starting tokenization of texts.")
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []

    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].to_list()
        logging.debug("Tokenizing chunk %d-%d", i, i + chunk_size)

        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    logging.info("Tokenization completed for %d texts.", len(texts))
    return np.array(all_ids)


def create_tf_datasets(
        X_train: np.array, y_train: np.array, 
        X_val: np.array, y_val: np.array, 
        X_test: np.array, y_test: np.array, params: dict) -> tuple:
    """
    Create TensorFlow datasets for training, validation and test set.

    Args:
        X_train (numpy.ndarray): Features for the training dataset.
        y_train (numpy.ndarray): Labels for the training dataset.
        X_val (numpy.ndarray): Features for the validation dataset.
        y_val (numpy.ndarray): Labels for the validation dataset.
        X_test (numpy.ndarray): Features for the test dataset.
        params (dict): Dictionary of parameters for preprocessing.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    logging.info("Creating TensorFlow datasets.")

    try: 
        AUTO = tf.data.AUTOTUNE if params["AUTO_PREFETCH"] else tf.data.Dataset.prefetch
        logging.info("Creating training dataset.")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(
            params["SHUFFLE_BUFFER"]).batch(params["BATCH_SIZE"]).prefetch(AUTO)

        logging.info("Creating validation dataset.")
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
            params["BATCH_SIZE"]).cache().prefetch(AUTO)
        
        logging.info("Creating test dataset.")
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            params["BATCH_SIZE"]).cache().prefetch(AUTO)

        logging.info("Datasets creation completed successfully.")
        return np.array([train_dataset]), np.array([val_dataset]), np.array([test_dataset])
    except Exception as e:
        logging.error(f"Error in creating Tensorflow datasets: {e}")