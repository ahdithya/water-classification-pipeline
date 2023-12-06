"""Transform module for components in the pipeline.
"""

import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = {
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
}

LABEL_KEY = "Potability"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs (dict): map from feature keys to raw not-yet-transformed features.

    Returns:
        outputs (dict): map from string feature key to transformed feature operations.
    """
    outputs = {}

    for feature in NUMERICAL_FEATURES:
        scaled = tft.scale_to_0_1(inputs[feature])
        outputs[transformed_name(feature)] = tf.cast(scaled, tf.float64)

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
