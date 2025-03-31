

# tfx_pipeline/pipelines/query_pipeline.py

import os
from typing import List

from tfx import v1 as tfx

def create_pipeline(pipeline_name: str,
                    pipeline_root: str,
                    data_path: str,
                    module_file: str,
                    metadata_path: str) -> tfx.dsl.Pipeline:
    # CsvExampleGen
    example_gen = tfx.components.CsvExampleGen(input_base=data_path)

    # StatisticsGen
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # SchemaGen
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    # ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Transform
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file)

    components: List[tfx.dsl.components.base.base_component.BaseComponent] = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=[
            "--direct_running_mode=multi_processing",
            "--direct_num_workers=0",
        ]
    )

