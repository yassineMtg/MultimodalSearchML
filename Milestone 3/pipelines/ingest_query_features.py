import os
import tfx
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import pipeline
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
from tfx.orchestration.local import local_dag_runner
from tfx.proto import example_gen_pb2 #, transform_pb2
from tfx.components import Transform
from ingestion import preprocessing


def create_pipeline(pipeline_name: str,
                    pipeline_root: str,
                    data_path: str,
                    metadata_path: str):
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='single_split', pattern='query_features_flat.csv')
    ])
    example_gen = CsvExampleGen(input_base=data_path, input_config=input_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath("ingestion/preprocessing.py"),
        custom_config={"exclude_features": ["query_id"]},
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=[
            "--direct_running_mode=multi_processing",
            "--direct_num_workers=0"
        ]
    )
