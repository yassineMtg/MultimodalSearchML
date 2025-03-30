
import os
import tfx
from tfx.orchestration import pipeline
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator, Transform
from tfx.orchestration.local import local_dag_runner
from tfx_pipeline.components.ingest_query_features import create_query_example_gen_component

def create_pipeline(pipeline_name: str,
                    pipeline_root: str,
                    data_path: str,
                    metadata_path: str):
    # Step 1 - Ingestion
    example_gen = create_query_example_gen_component(data_path=data_path)
    
    # Step 2 - Statistics Generation
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    
    # Step 3 - Schema Generation
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    
    # Step 4 - Anomaly Detection
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Step 5 - Preprocessing / Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(os.path.dirname(__file__), "../scripts/preprocessing.py"),
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

