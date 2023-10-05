import tensorflow_datasets as tfds

import keras_nlp
import keras_core as keras

keras.mixed_precision.set_global_policy('mixed_float16')

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
imdb_train = imdb_train.map(lambda x, y: x)
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)
gpt2_lm.summary()
# Fine-tune on IMDb movie reviews.
gpt2_lm.fit(imdb_train)
