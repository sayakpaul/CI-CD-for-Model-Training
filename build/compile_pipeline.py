import argparse

from absl import logging
from create_pipeline import create_pipeline

from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

import os
import sys
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from utils import config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-gpu",
        type=str,
        required=False,
        default="False"
    )

    return parser.parse_args()


def compile_pipeline(args):
    pipeline_definition_file = config.PIPELINE_NAME + ".json"

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
                default_image=config.TFX_IMAGE_URI
            ),
        output_filename=pipeline_definition_file,
    )

    use_gpu = True if args.use_gpu == "True" else False

    return runner.run(
        create_pipeline(
            num_epochs=data_types.RuntimeParameter(name="num_epochs", ptype=int),
            learning_rate=data_types.RuntimeParameter(name="learning_rate", ptype=float),
            use_gpu=use_gpu,
        ),
        write_out=True,
    )


def main():
    args = get_args()
    result = compile_pipeline(args)
    logging.info(result)


if __name__ == "__main__":
    main()
