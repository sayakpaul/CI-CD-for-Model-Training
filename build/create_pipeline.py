from tfx import v1 as tfx


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    serving_model_dir: str,
    project_id: str,
    region: str,
    use_gpu: bool,
) -> tfx.dsl.Pipeline:
    """Implements the penguin pipeline with TFX."""
    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # NEW: Configuration for Vertex AI Training.
    # This dictionary will be passed as `CustomJobSpec`.
    vertex_job_spec = {
        "project": project_id,
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
        module_file=module_file,
        examples=example_gen.outputs["examples"],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_UCAIP_KEY: True,
            tfx.extensions.google_cloud_ai_platform.UCAIP_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec,
            "use_gpu": use_gpu,
        },
    )

    # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = [
        example_gen,
        trainer,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name, pipeline_root=pipeline_root, components=components
    )
