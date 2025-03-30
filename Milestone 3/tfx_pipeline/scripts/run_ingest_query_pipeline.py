
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from tfx_pipeline.pipelines.query_pipeline import create_pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# pipeline configs
PIPELINE_NAME = "multimodal_search_pipeline"
DATA_PATH = os.path.join("data", "raw")
PIPELINE_ROOT = os.path.join("artifacts", PIPELINE_NAME)
METADATA_PATH = os.path.join("tfx_metadata", "metadata.db")

if __name__ == "__main__":
    print("🚀 Starting Multimodal Search TFX pipeline...")

    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_path=DATA_PATH,
        metadata_path=METADATA_PATH,
    )

    LocalDagRunner().run(pipeline)

    print("✅ Pipeline finished successfully!")

