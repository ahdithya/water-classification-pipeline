"""This module contains code for tuning the model using keras-tuner.
"""
from typing import Any, Dict, NamedTuple, Text
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
import keras_tuner as kt
from keras_tuner.engine import base_tuner
from transform import NUMERICAL_FEATURES, transformed_name
from trainer import input_fn

TunerFnResult = NamedTuple(
    "TunerFnResult",
    [
        ("tuner", base_tuner.BaseTuner),
        ("fit_kwargs", Dict[Text, Any]),
    ],
)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)


def get_tuned_model(hp):
    """
    Create a Keras model with hyperparameters to be tuned.

    Args:
        hp (keras_tuner.HyperParameters): HyperParameters object.

    Returns:
        tf.keras.Model: a compiled Keras model
    """
    input_features = []

    # Add numerical features
    for feature in NUMERICAL_FEATURES:
        input_features.append(layers.Input(shape=(1,), name=transformed_name(feature)))

    concatenated_inputs = layers.Concatenate()(input_features)
    x = layers.Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice(
            "dense_activation",
            values=["relu", "tanh", "sigmoid"],
            default="relu",
        ),
    )(concatenated_inputs)
    x = layers.Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice(
            "dense_activation",
            values=["relu", "tanh", "sigmoid"],
            default="relu",
        ),
    )(x)
    x = layers.Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice(
            "dense_activation",
            values=["relu", "tanh", "sigmoid"],
            default="relu",
        ),
    )(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model


def tuner_fn(fn_args):
    """Tuning the model to get the best hyperparameters based on given args

    Args:
        fn_args (FnArgs): Holds args used to train the model as name/value pair

    Returns:
        TunerFnResult (NamedTuple): object to run model tuner
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files[0],
        tf_transform_output,
    )
    eval_set = input_fn(
        fn_args.eval_files[0],
        tf_transform_output,
    )

    tuner = kt.Hyperband(
        hypermodel=get_tuned_model,
        objective=kt.Objective(
            "val_loss",
            direction="min",
        ),
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "callbacks": [early_stop],
        },
    )
