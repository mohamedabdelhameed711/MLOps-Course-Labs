import os
import logging
import joblib
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/gb_model.pkl")  

@lru_cache(maxsize=1)
def load_model():
    logger.info("Loading model from local file: %s", MODEL_PATH)

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise
