"""Trainer module for training a Keras model"""

import os
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from transform import LABEL_KEY, NUMERICAL_FEATURES, transformed_name


def gzip_reader_fn(filenames):
    """Load compressed tfrecord files"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Create tf.data.Dataset object from tfrecord files

    Args:
        file_pattern (str): a path to the tfrecord files
        tf_transform_output (tft.TFTransformOutput): a transform output
        batch_size (int): batch size. Defaults to 64.

    Returns:
        tf.data.Dataset: a dataset object

    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def get_model():
    """Create a Keras model with keras.layers"""
    input_features = []

    # Add numerical features
    for feature in NUMERICAL_FEATURES:
        input_features.append(layers.Input(
            shape=(1,), name=transformed_name(feature)))

    concatenated_inputs = layers.Concatenate()(input_features)
    x = layers.Dense(256, activation="relu")(concatenated_inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Return a function that parses a serialized tf.Example"""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"),
        ]
    )
    def serve_tf_examples_fn(serialized_tf_examples):
        """Return the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    """Train the model based on given args

    Args:
        fn_args (trainer.TrainerFnArgs): Holds args used to train the model as name/value pairs.

    Returns:
        None
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Use transformed_examples from the transform component output
    train_set = input_fn(fn_args.train_files, tf_transform_output)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", verbose=1, patience=10
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    model.fit(
        train_set,
        steps_per_epoch=100,
        validation_data=eval_set,
        validation_steps=32,
        callbacks=[tensorboard_callback, es, mc],
        epochs=30,
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model,
            tf_transform_output,
        )
    }

    model.save(fn_args.serving_model_dir,
               save_format="tf", signatures=signatures)
