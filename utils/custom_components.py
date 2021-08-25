"""
Taken from:
    * https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai/blob/main/src/tfx_pipelines/components.py#L51
"""

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import (
    InputArtifact,
    OutputArtifact,
    Parameter,
)
from tfx.types.standard_artifacts import HyperParameters
from tfx.types import artifact_utils
from tfx.utils import io_utils
import logging
import json
import os


@component
def hyperparameters_gen(
    num_epochs: Parameter[int],
    learning_rate: Parameter[float],
    hyperparameters: OutputArtifact[HyperParameters],
):

    hp_dict = dict()
    hp_dict["num_epochs"] = num_epochs
    hp_dict["learning_rate"] = learning_rate
    logging.info(f"Hyperparameters: {hp_dict}")

    hyperparams_uri = os.path.join(
        artifact_utils.get_single_uri([hyperparameters]), "hyperparameters.json"
    )
    io_utils.write_string_file(hyperparams_uri, json.dumps(hp_dict))
    logging.info(f"Hyperparameters are written to: {hyperparams_uri}")
