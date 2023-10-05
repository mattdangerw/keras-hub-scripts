import keras_nlp
import numpy as np

input_data = {
    "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
}

# Pretrained BERT encoder.
model = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")
model(input_data)

features = {
    "token_ids": np.ones(shape=(2, 12), dtype="int32"),
    "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
}
labels = [0, 3]

# Pretrained classifier without preprocessing.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=4,
    preprocessor=None,
)
classifier.fit(x=features, y=labels, batch_size=2)
