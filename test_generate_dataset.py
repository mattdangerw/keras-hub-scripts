import tensorflow as tf

import keras_nlp

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
gpt2_lm.compile(sampler="greedy")

# Raw inputs.
prompt = ["that's weird", "that's even weirder"]
print(gpt2_lm.generate(prompt, max_length=32))

# Dataset inputs inputs.
dataset = tf.data.Dataset.from_tensor_slices(prompt * 25).batch(2)
print(gpt2_lm.generate(dataset, max_length=32))

# Preprocessed inputs.
preprocessor = keras_nlp.models.GPT2Preprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=20,
    add_start_token=True,
)
gpt2_lm.preprocessor = None

preprocessed_prompt = preprocessor(prompt)
print(preprocessed_prompt)
print(gpt2_lm.generate(preprocessed_prompt, max_length=32))
