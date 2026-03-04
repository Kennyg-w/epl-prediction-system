import joblib
from .config import MODEL_PATH

import joblib
from .config import MODEL_PATH

def load_artifact(which: str = "raw"):
    bundle = joblib.load(MODEL_PATH)

    features = bundle.get("features", None)

    if which == "calibrated":
        return bundle["model_calibrated"], bundle["threshold_calibrated"], features

    return bundle["model_raw"], bundle["threshold_raw"], features