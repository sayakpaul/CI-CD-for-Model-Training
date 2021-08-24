import os

# GCP
GCP_PROJECT = os.getenv("PROJECT")
GCP_REGION = os.getenv("REGION")

# Data
DATA_ROOT = os.getenv("DATA_ROOT")

# Training and serving
MODULE_ROOT = os.getenv("MODULE_ROOT")
MODULE_FILE = os.path.join(MODULE_ROOT + "penguin_trainer.py")
SERVING_MODEL_DIR = os.getenv("MODULE_ROOT")

# Pipeline
PIPELINE_NAME = os.getenv("PIPELINE_NAME")
PIPELINE_ROOT = os.getenv("SERVING_MODEL_DIR")
