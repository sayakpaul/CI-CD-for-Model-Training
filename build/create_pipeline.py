from tfx.orchestration import data_types
from tfx import v1 as tfx

import os
import sys
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from utils import config, custom_components

def create_pipeline(
    num_epochs: data_types.RuntimeParameter,
    batch_size: data_types.RuntimeParameter,
    optimizer: data_types.RuntimeParameter,
    use_gpu: bool,
) -> tfx.dsl.Pipeline:
    """Implements the penguin pipeline with TFX."""
    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=config.DATA_ROOT)

    # Generate hyperparameters.
    hyperparams_gen = custom_components.hyperparameters_gen(
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer
    )

    # NEW: Configuration for Vertex AI Training.
    # This dictionary will be passed as `CustomJobSpec`.
    vertex_job_spec = {
        "project": config.GCP_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "gcr.io/tfx-oss-public/tfx:{}".format(tfx.__version__),
                },
            }
        ],
    }
    if use_gpu:
        # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
        # for available machine types.
        vertex_job_spec["worker_pool_specs"][0]["machine_spec"].update(
            {"accelerator_type": "NVIDIA_TESLA_K80", "accelerator_count": 1}
        )

    # Trains a model using Vertex AI Training.
    # NEW: We need to specify a Trainer for GCP with related configs.
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=config.MODULE_FILE,
        examples=example_gen.outputs["examples"],
        train_args=tfx.proto.TrainArgs(num_steps=0),
        eval_args=tfx.proto.EvalArgs(num_steps=None),
        hyperparameters=hyperparams_gen.outputs.hyperparameters,
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY: True,
            tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY: config.GCP_REGION,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec,
            "use_gpu": use_gpu,
        },
    )

    # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=config.SERVING_MODEL_DIR
            )
        ),
    )

    components = [
        example_gen,
        hyperparams_gen,
        trainer,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=config.PIPELINE_NAME, pipeline_root=config.PIPELINE_ROOT, components=components
    )
