import yaml
from .htm_model import HTMModel
from encoder import ScalarEncoder


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    feature_encoders = {}
    for feat, params in config["features"].items():
        feature_encoders[feat] = ScalarEncoder(
            n=params["n"],
            w=params["w"],
            min_val=params["minval"],
            max_val=params["maxval"]
        )

    shared_model = config.get("shared_model", False)
    tm_args = config.get("temporal_memory", {})

    model = HTMModel(
        encoders=feature_encoders,
        shared_model=shared_model,
        tm_args=tm_args
    )
    return model