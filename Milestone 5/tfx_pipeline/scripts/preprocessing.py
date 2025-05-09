import tensorflow as tf
import tensorflow_transform as tft

# def preprocessing_fn(inputs):
#     outputs = {}

#     # Pass through all 768 numerical features
#     for i in range(768):
#         outputs[f"f{i}"] = inputs[f"f{i}"]

#     # Pass the real label directly
#     outputs["label"] = inputs["label"]

#     return outputs

def preprocessing_fn(inputs):
    outputs = {}

    for i in range(768):
        outputs[f"f{i}"] = inputs[f"f{i}"]

    outputs["label"] = inputs["label"]  # Must be present
    return outputs

