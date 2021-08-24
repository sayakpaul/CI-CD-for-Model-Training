import os

# GCP
GCP_PROJECT = os.getenv("_PROJECT")
GCP_REGION = os.getenv("_REGION")

# Data
DATA_ROOT = os.getenv("_DATA_ROOT")

# Training and serving
MODULE_ROOT = os.getenv("_MODULE_ROOT")
MODULE_FILE = os.path.join(MODULE_ROOT + "penguin_trainer.py")
SERVING_MODEL_DIR = os.getenv("_MODULE_ROOT")

# Pipeline
PIPELINE_NAME = os.getenv("_PIPELINE_NAME")
PIPELINE_ROOT = os.getenv("_SERVING_MODEL_DIR")
