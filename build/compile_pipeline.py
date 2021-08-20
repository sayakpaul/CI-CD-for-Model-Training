
import os
import tfx
import logging
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project',  
        type=str,
    )

    parser.add_argument(
        '--region', 
        type=str,
    )

    parser.add_argument(
        '--pipeline-name', 
        type=str,
    )

    parser.add_argument(
        '--pipeline-root', 
        type=str,
    )

    parser.add_argument(
        '--data-root', 
        type=str,
    )

    parser.add_argument(
        '--module-root', 
        type=str,
    )

    parser.add_argument(
        '--serving-model-dir', 
        type=str,
    )
    
    return parser.parse_args()


def compile_pipeline(args):
    PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=args.pipeline_definition_file)
    return runner.run(
        create_pipeline(
            pipeline_name=args.pipeline_name,
            pipeline_root=args.pipeline_root,
            data_root=args.data_root,
            module_file=os.path.join(args.module_root + "penguin_trainer.py"),
            serving_model_dir=args.serving_model_dir,
            project_id=args.project,
            region=args.region,
            # We will use CPUs only for now.
            use_gpu=False
        ),
        write_out=True
    )

def main():
    args = get_args()
    result = compile_pipeline(args)
    logging.info(result)

if __name__ == "__main__":
    main()
