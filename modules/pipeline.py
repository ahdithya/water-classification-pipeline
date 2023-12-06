""" Defines the pipeline  """

from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline


def init_pipeline(
    components, pipeline_root: Text, metadata_path, pipeline_name
) -> pipeline.Pipeline:
    """
    Initiate the pipeline

    Args:
        components (list): list of TFX components
        pipeline_root (Text): path to the pipeline root
        metadata_path ([type]): path to the metadata
        pipeline_name ([type]): name of the pipeline

    Returns:
        pipeline.Pipeline
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args,
    )
