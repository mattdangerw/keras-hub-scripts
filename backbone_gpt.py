import keras_nlp
import numpy as np
import keras_core as keras

keras.mixed_precision.set_dtype_policy("mixed_float16")

input_data = {
    "token_ids": np.array([[1169, 2068, 7586, 21831, 13]]),
    "padding_mask": np.array([[1, 1, 1, 1, 1]]),
}
model = keras_nlp.models.GPT2Backbone.from_preset("gpt2_base_en")
print(model.predict(input_data))
print(model.predict(input_data))
