import keras_nlp
from tensorflow import keras

opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
prompt = ["that's weird", "that's even weirder"]
# Eager generation.
opt_lm.compile(sampler="beam")
print(opt_lm.generate(prompt, max_length=30))
