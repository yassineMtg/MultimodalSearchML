
# tfx_pipeline/scripts/run_ingest_query_pipeline.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx_pipeline.pipelines.query_pipeline import create_pipeline

def run():
    # Dynamically detect root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pipeline_name = "multimodal_search_pipeline"
    pipeline_root = os.path.join(project_root, "artifacts")
    data_path = os.path.join(project_root, "../data", "query")
    module_file = os.path.join(project_root, "scripts", "preprocessing.py")
    metadata_path = os.path.join(project_root, "tfx_metadata", "metadata.db")

    print("🚀 Starting Multimodal Search TFX pipeline...")
    pipeline = create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_path=data_path,
        module_file=module_file,
        metadata_path=metadata_path,
    )
    LocalDagRunner().run(pipeline)
    print("✅ Pipeline finished successfully!")

if __name__ == "__main__":
    run()
