import yaml
from htm_py.htm_model import HTMModel

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_model_from_config(config):
    encoders = config['encoders']
    sp_params = config.get('sp_params', {})
    tm_params = config.get('tm_params', {})
    use_shared_model = config.get('use_shared_model', False)

    return HTMModel(
        encoders=encoders,
        sp_params=sp_params,
        tm_params=tm_params,
        use_shared_model=use_shared_model
    )
