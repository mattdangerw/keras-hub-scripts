import time

import keras_core as keras
import tensorflow as tf
import keras_nlp
import tensorflow_datasets as tfds

keras.mixed_precision.set_dtype_policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy("mixed_float16")

ds = tfds.load(
    "imdb_reviews",
    split="train",
    as_supervised=True,
    batch_size=4,
)
ds = ds.map(lambda x, y: x)

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
gpt2_lm.fit(ds)
prompt = ["that's weird", "that's even weirder"]

# First time to compile.
print(gpt2_lm.generate(prompt, max_length=256))

# Measure subsequent runs.
start = time.time()
for i in range(25):
    print(gpt2_lm.generate(prompt, max_length=256))
end = time.time()
print(end - start)
