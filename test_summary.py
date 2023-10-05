import keras_nlp

# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=2,
)
classifier.summary(show_trainable=True)
