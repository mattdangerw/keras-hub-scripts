import keras_nlp
import keras_core as keras
import numpy as np

batch_size, feature_size = 4, 16
rank = 4
inputs = np.random.uniform(size=(batch_size, feature_size))
inner_dense = keras.layers.Dense(feature_size)
lora_dense = keras_nlp.layers.LoraDense(inner_dense, rank=4)
# Output with inner dense begins equal.
assert np.allclose(inner_dense(inputs), lora_dense(inputs))

# Add some random updates to the lora parameters.
lora_dense.lora_a.assign(np.random.uniform(size=(feature_size, rank)))
lora_dense.lora_b.assign(np.random.uniform(size=(rank, feature_size)))
assert not np.allclose(inner_dense(inputs), lora_dense(inputs))

# Merge the lora dense and output
lora_dense.merge_weights()
assert np.allclose(inner_dense(inputs), lora_dense(inputs))


batch_size, sequence_length, feature_size = 4, 10, 16
num_heads = 2
rank = 4
inputs = np.random.uniform(size=(batch_size, sequence_length, feature_size))
inner_dense = keras.layers.EinsumDense(
    "abc,cde->abde",
    output_shape=(sequence_length, num_heads, feature_size // num_heads),
)
lora_dense = keras_nlp.layers.LoraDense(inner_dense, rank=4)
# Output shape (4, 10, 2, 8)
lora_dense(inputs)