import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.pipeline import Pipeline
from tfx.proto import trainer_pb2
from tfx.components import CsvExampleGen, Trainer
from tfx.components import Transform
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.orchestration.metadata import sqlite_metadata_connection_config

# Paths
_pipeline_name = "training_pipeline"
_pipeline_root = os.path.join("tfx_pipeline", "artifacts", _pipeline_name)
data_root = os.path.abspath("data/processed")  # folder where merged_queries.csv is stored
module_file = os.path.abspath("tfx_pipeline/components/trainer.py")
transform_module_file = os.path.abspath("tfx_pipeline/scripts/preprocessing.py")
_serving_model_dir = os.path.join(_pipeline_root, "serving_model")
_metadata_path = os.path.join("tfx_pipeline", "tfx_metadata", f"{_pipeline_name}.sqlite")

# Pipeline definition
def create_pipeline():
    example_gen = CsvExampleGen(input_base=data_root)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    trainer = Trainer(
        run_fn='tfx_pipeline.components.trainer.run_fn',
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=200),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
        custom_config={"model_run_dir": _pipeline_root}
    )

    return Pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer
        ],
        enable_cache=True,
        metadata_connection_config=sqlite_metadata_connection_config(_metadata_path),
        beam_pipeline_args=[],
    )

if __name__ == '__main__':
    LocalDagRunner().run(create_pipeline())
