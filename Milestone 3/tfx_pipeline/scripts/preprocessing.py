# tfx_pipeline/scripts/preprocessing.py

import tensorflow as tf
import tensorflow_transform as tft

def _split_and_parse(tensor):
    """
    Helper function to clean, split and parse string tensors to float tensors.
    """
    cleaned = tf.strings.regex_replace(tensor, r"[\[\]\n]", "")
    split = tf.strings.split(cleaned)
    return tf.strings.to_number(split, out_type=tf.float32)

def preprocessing_fn(inputs):
    """
    The preprocessing function for query features.

    This function keeps the 768 numerical features and passes them unchanged.
    If required, additional preprocessing like normalization can be added.
    """
    outputs = {}

    for i in range(768):
        feature_name = f"f{i}"
        outputs[feature_name] = inputs[feature_name]

    # Handle event_timestamp correctly if needed later

    return outputs
