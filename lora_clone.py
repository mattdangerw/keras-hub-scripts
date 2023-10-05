import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_nlp
import tensorflow_datasets as tfds
import keras_core as keras
import re


def add_lora_layers(model, targets=[".*query", ".*value"]):
    class LoraLayerWrapper(keras.layers.Layer):
        def add_weight(self, *args, **kwargs):
            super().add_weights(*args, **kwargs)

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                inject_wrapper(value.__class__)
                with keras.name_scope(value.name, caller=value):
                    path = keras.src.backend.common.name_scope.current_path()
                if any(re.matches(target, path) for target in targets):
                    value = keras.layers.LoraDense(value)
            return super().__setattr__(name, value)

    def inject_wrapper(cls):
        if cls == keras.layers.Layer:
            return LoraLayerWrapper
        if cls == object:
            return object
        cls.__bases__ = tuple(inject_wrapper(base) for base in cls.__bases__)
        cls.__new__(cls)
        return cls

    cls = model.__class__
    cls = inject_wrapper(cls)
    return cls.from_config(model.get_config())

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
backbone = keras_nlp.models.BartBackbone.from_preset(
    "bert_base_en_uncased",
)
lora_backbone = add_lora_layers(backbone)
preprocessor = keras_nlp.models.BartPreprocessor.from_preset(
    "bert_base_en_uncased",
)
classifier = keras_nlp.models.BertClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=2,
)
classifier.summary()
classifier.fit(imdb_train)
classifier.evaluate(imdb_test)
