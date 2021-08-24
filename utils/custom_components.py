"""
References:
    * https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai/blob/main/src/tfx_pipelines/components.py#L51
"""

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import (
    InputArtifact,
    OutputArtifact,
    Parameter,
)
from tfx.types.standard_artifacts import HyperParameters
import logging


@component
def hyperparameters_gen(
    num_epochs: Parameter[int],
    batch_size: Parameter[int],
    optimizer: Parameter[str],
    hyperparameters: OutputArtifact[HyperParameters],
):

    hp_dict = dict()
    hp_dict["num_epochs"] = num_epochs
    hp_dict["batch_size"] = batch_size
    hp_dict["optimizer"] = optimizer
    logging.info(f"Hyperparameters: {hp_dict}")
