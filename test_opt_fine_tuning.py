import tensorflow_datasets as tfds
import tensorflow as tf

import keras_nlp

# imdb_train = tfds.load(
#     "imdb_reviews",
#     split="train",
#     as_supervised=True,
#     batch_size=2,
# ).map(lambda x, y: x)

ds = tf.data.Dataset.from_tensor_slices(["this is a test"] * 1000).batch(8)

opt_lm_preprocessor = keras_nlp.models.OPTCausalLMPreprocessor.from_preset(
    "opt_125m_en",
    sequence_length=32,
)
opt_lm = keras_nlp.models.OPTCausalLM.from_preset(
    "opt_125m_en",
    preprocessor=opt_lm_preprocessor,
)
opt_lm.run_eagerly = True
opt_lm.fit(ds, steps_per_epoch=5)
print(opt_lm.generate(["this", "this"]))
