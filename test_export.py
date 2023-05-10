import tensorflow as tf
from tensorflow import keras

import keras_nlp

# Keep preprocessing short for demo.
gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=30,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en",
    preprocessor=gpt2_preprocessor,
)


def export_fn(x):
    x = gpt2_preprocessor.generate_preprocess(x)
    x = gpt2_lm.make_generate_function()(x)
    return gpt2_preprocessor.generate_postprocess(x)


export_archive = keras.export.ExportArchive()
export_archive.track(gpt2_lm)
export_archive.add_endpoint(
    name="serve",
    fn=export_fn,
    input_signature=[tf.TensorSpec(shape=(None), dtype=tf.string)],
)
export_archive.write_out("path/to/location")

serving_model = tf.saved_model.load("path/to/location")
outputs = serving_model.serve(tf.constant(["The movie was bad"]))
