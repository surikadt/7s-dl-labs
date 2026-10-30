import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def save_model(model, prefix):
    prefix = Path(prefix)
    config = json.loads(model.to_json())
    config.pop('compile_config', None)
    prefix.with_suffix('.json').write_text(json.dumps(config, indent=2))
    np.savez_compressed(prefix.with_suffix('.npz'), *model.get_weights())


def load_model(prefix):
    prefix = Path(prefix)
    model = keras.models.model_from_json(prefix.with_suffix('.json').read_text())
    with np.load(prefix.with_suffix('.npz')) as arrays:
        model.set_weights([arrays[f'arr_{i}'] for i in range(len(arrays.files))])
    return model
