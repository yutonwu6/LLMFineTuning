import os
import os.path as op


class BaseConfig(object):
    BASEDIR = op.abspath(op.dirname(__file__))
    PROJECT_ROOT = BASEDIR

    MODEL_NAME = "./Qwen3-0.6B"

    FINE_TUNING = "qlora"

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    EPOCHS = 1
    BATCH_SIZE = 4
    MAX_STEPS = 30
    LEARNING_RATE = 2e-5
    MAX_OUTPUT_TOKEN = 200
    TOP_P = 0.9
    TEMPERATURE = 0.7

    MAX_LENGTH = 512

    OUTPUT_DIR = "./saved_models"

Config = BaseConfig