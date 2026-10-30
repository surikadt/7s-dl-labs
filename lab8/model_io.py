import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(name, size=150, units=128, dropout=0.5, extra_units=0):
    constructors = {'vgg16': keras.applications.VGG16,
                    'resnet50': keras.applications.ResNet50,
                    'inception_v3': keras.applications.InceptionV3}
    base = constructors[name](weights='imagenet', include_top=False,
                              input_shape=(size, size, 3), pooling='avg')
    base.trainable = False
    head = keras.Sequential(name='head')
    head.add(layers.Input(shape=base.output_shape[1:]))
    head.add(layers.Dense(units, activation='relu'))
    head.add(layers.Dropout(dropout))
    if extra_units:
        head.add(layers.Dense(extra_units, activation='relu'))
    head.add(layers.Dense(1, activation='sigmoid'))
    inputs = keras.Input(shape=(size, size, 3))
    outputs = head(base(inputs, training=False))
    model = keras.Model(inputs, outputs, name=name)
    return model, base, head


def save_bundle(model, base, head, prefix, config):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Store the trained head and changed base layers; ImageNet supplies the remaining weights.
    arrays = {}
    names = []
    for index, layer in enumerate(base.layers):
        if layer.trainable and layer.weights:
            names.append({'index': index, 'name': layer.name})
            for i, value in enumerate(layer.get_weights()):
                arrays[f'base_{index}_{i}'] = value
    for i, value in enumerate(head.get_weights()):
        arrays[f'head_{i}'] = value
    np.savez_compressed(prefix.with_suffix('.npz'), **arrays)
    metadata = {**config, 'base_layers': names, 'head_weights': len(head.get_weights())}
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2))


def load_bundle(prefix):
    prefix = Path(prefix)
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    config = {key: metadata[key] for key in ['name', 'size', 'units', 'dropout', 'extra_units']}
    model, base, head = build_model(**config)
    with np.load(prefix.with_suffix('.npz')) as arrays:
        for record in metadata['base_layers']:
            index = record['index']
            layer = base.layers[index]
            values = [arrays[f'base_{index}_{i}'] for i in range(len(layer.weights))]
            layer.set_weights(values)
        head.set_weights([arrays[f'head_{i}'] for i in range(metadata['head_weights'])])
    return model
