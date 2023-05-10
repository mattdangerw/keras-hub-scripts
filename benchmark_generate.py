import time

import keras_nlp

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
gpt2_lm.compile(sampler="greedy")
prompt = ["that's weird", "that's even weirder"]

# First time to compile.
print(gpt2_lm.generate(prompt, max_length=256))

# Measure subsequent runs.
start = time.time()
for i in range(25):
    gpt2_lm.generate(prompt, max_length=256)
end = time.time()
print(end - start)
