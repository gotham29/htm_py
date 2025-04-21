import yaml
from .htm_model import HTMModel
from htm_py.encoders import ScalarEncoder


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    encoder_params = {}
    for feat, params in config["features"].items():
        encoder_params[feat] = ScalarEncoder(
            n=params["n"],
            w=params["w"],
            minval=params["minval"],
            maxval=params["maxval"]
        )

    shared_model = config.get("shared_model", False)
    tm_args = config.get("temporal_memory", {})

    model_type = config.get("model_type", "shared")
    sp_params = config.get("sp_params", {})
    tm_params = config.get("tm_params", {})

    model = HTMModel(
        encoders=encoder_params,
        sp_params=sp_params,
        tm_params=tm_params,
        model_type=model_type,
    )

    return model
