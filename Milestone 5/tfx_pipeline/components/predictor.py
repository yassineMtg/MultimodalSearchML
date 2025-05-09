import os
from typing import Any, Dict

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.base import base_component, base_executor
from tfx.dsl.components.base.base_executor import BaseExecutor
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ComponentSpec

class PredictorExecutor(BaseExecutor):
    def Do(self, input_dict: Dict[str, Any], output_dict: Dict[str, Any], exec_properties: Dict[str, Any]) -> None:
        transform_graph = input_dict['transform_graph'][0]
        examples = input_dict['examples'][0]

        tf_transform_output = tft.TFTransformOutput(transform_graph.uri)
        dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(os.path.join(examples.uri, '*')), compression_type='GZIP')
        parsed = dataset.map(lambda x: tf.io.parse_single_example(x, tf_transform_output.transformed_feature_spec()))

        for example in parsed.take(1):
            print("✅ Sample transformed example ready for inference.")


class PredictorSpec(ComponentSpec):
    PARAMETERS = {}
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples),
        'transform_graph': ChannelParameter(type=standard_artifacts.TransformGraph),
    }
    OUTPUTS = {}

class Predictor(base_component.BaseComponent):
    SPEC_CLASS = PredictorSpec
    EXECUTOR_SPEC = base_component.ExecutorClassSpec(PredictorExecutor)

    def __init__(self, examples, transform_graph):
        spec = PredictorSpec(examples=examples, transform_graph=transform_graph)
        super().__init__(spec=spec)
