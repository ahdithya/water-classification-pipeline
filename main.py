""" Running the pipeline."""
import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules import pipeline, components

PIPELINE_NAME = "water-classification-pipeline"

# pipeline input
DATA_ROOT = "data"
TRANSFORM_MODULE = "modules/transform.py"
TUNER_MODULE = "modules/tuner.py"
TRAINER_MODULE = "modules/trainer.py"


# pipeline output
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model")
pipeline_root = os.path.join(OUTPUT_BASE, "pipelines", PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata")

components_args = {
    "data_dir": DATA_ROOT,
    "transform_module": TRANSFORM_MODULE,
    "tuning_module": TUNER_MODULE,
    "training_module": TRAINER_MODULE,
    "training_steps": 5000,
    "eval_steps": 1000,
    "serving_model_dir": serving_model_dir,
}

components = components.init_components(components_args)

pipeline = pipeline.init_pipeline(
    components, pipeline_root, metadata_path, pipeline_name=PIPELINE_NAME
)
BeamDagRunner().run(pipeline=pipeline)
