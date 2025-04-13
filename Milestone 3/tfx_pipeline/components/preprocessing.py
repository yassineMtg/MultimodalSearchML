# tfx_pipeline/scripts/preprocessing.py

import tensorflow as tf
import tensorflow_transform as tft

def _split_and_parse(tensor):
    cleaned = tf.strings.regex_replace(tensor, r"[\[\]\n]", "")
    split = tf.strings.split(cleaned)
    return tf.strings.to_number(split, out_type=tf.float32)

NUM_FEATURES = 100 

def preprocessing_fn(inputs):
    outputs = {}

    for i in range(NUM_FEATURES):
        key = f"f{i}"
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # outputs["label"] = tf.cast(inputs["f0"] > 0.0, tf.int64)
    outputs["label"] = tf.cast(tf.greater(inputs["f0"], 0.0), tf.int64)
    return outputs
