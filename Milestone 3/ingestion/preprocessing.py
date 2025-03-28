import tensorflow as tf

def _split_and_parse(tensor):
    cleaned = tf.strings.regex_replace(tensor, r"[\[\]\n]", "")
    split = tf.strings.split(cleaned)
    return tf.strings.to_number(split, out_type=tf.float32)

def preprocessing_fn(inputs):
    # Only return the 768 float features
    return {
        f"f{i}": inputs[f"f{i}"] for i in range(768)
    }

