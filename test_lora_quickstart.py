import tensorflow_datasets as tfds

import keras_nlp
import keras_core as keras

keras.mixed_precision.set_global_policy("mixed_float16")

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
)
classifier.summary()
classifier.add_lora_layers(
    trainable_weight_paths=[".*bias"],
)
classifier.summary()
classifier.fit(imdb_train, validation_data=imdb_test)
classifier.evaluate(imdb_test)
classifier.merge_lora_layers()
classifier.evaluate(imdb_test)
classifier.save("test.keras")

# TODO merge won't work on jax! There is no variables just arrays.
# We would need a get_weights/set_weights? Perhaps?


# gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
# gpt2_lm.preprocessor.packer.sequence_length = 128
# gpt2_lm.backbone.add_lora_layers()
# gpt2_lm.summary()
# gpt2_lm.fit(imdb_train.map(lambda x, y: x))
# gpt2_lm.backbone.merge_lora_layers()
# gpt2_lm.generate(["that's weird", "that's even weirder"], max_length=32)
