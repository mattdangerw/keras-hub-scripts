import time

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM

# 1. Load model and tokenizer
model_name = "gpt2"
# remember: decoder-only models need left-padding
tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding_side="left", pad_token="</s>"
)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# 2. Prepare tokenization and generation arguments -- don't forget padding to avoid retracing!
tokenization_kwargs = {
    "pad_to_multiple_of": 256,
    "padding": True,
    "return_tensors": "tf",
}
generation_kwargs = {"penalty_alpha": 0.6, "top_k": 4, "max_new_tokens": 256}

# 3. Create your XLA generate function a̶n̶d̶ ̶m̶a̶k̶e̶ ̶P̶y̶T̶o̶r̶c̶h̶ ̶e̶a̶t̶ ̶d̶u̶s̶t̶
# This is the only change with respect to original generate workflow!
xla_generate = tf.function(model.generate, jit_compile=True)

# 4. Generate! Remember -- the first call will be slow, but all subsequent calls will be fast if you've done things right.
input_prompts = ["that's weird", "that's even weirder"]
tokenized_inputs = tokenizer(input_prompts, **tokenization_kwargs)
generated_text = xla_generate(**tokenized_inputs, **generation_kwargs)
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(decoded_text)
decoded_text = tokenizer.decode(generated_text[1], skip_special_tokens=True)
print(decoded_text)

# Measure subsequent runs.
start = time.time()
for i in range(25):
    input_prompts = ["that's weird", "that's even weirder"]
    tokenized_inputs = tokenizer(input_prompts, **tokenization_kwargs)
    generated_text = xla_generate(**tokenized_inputs, **generation_kwargs)
    decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    decoded_text = tokenizer.decode(generated_text[1], skip_special_tokens=True)
end = time.time()
print(end - start)
