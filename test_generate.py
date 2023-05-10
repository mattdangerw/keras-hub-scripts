import keras_nlp

opt_lm = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
prompt = ["that's weird", "that's even weirder"]
# Eager generation.
opt_lm.compile(run_eagerly=True, sampler="contrastive")
print(opt_lm.generate(prompt, max_length=30))
# XLA generation.
opt_lm.compile(sampler="contrastive")
print(opt_lm.generate(prompt, max_length=30))
# Non-XLA generation.
opt_lm.compile(sampler="contrastive", jit_compile=False)
print(opt_lm.generate(prompt, max_length=30))
