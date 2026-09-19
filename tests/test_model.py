import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50


INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 2


def build_test_model():
    """
    Build the same architecture used in the project,
    but without downloading ImageNet weights.

    This makes the test fast and reproducible.
    """

    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    model = Sequential([
        base_model,
        Flatten(),
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    # Same transfer-learning strategy as the project
    for layer in base_model.layers[:-1]:
        layer.trainable = False

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def test_model_can_be_created():
    """
    Verify that the ResNet50-based model can be
    constructed successfully.
    """

    model = build_test_model()

    assert model is not None


def test_model_output_shape():
    """
    Verify that the model produces one probability
    for each target class.
    """

    model = build_test_model()

    assert model.output_shape == (
        None,
        NUM_CLASSES
    )


def test_model_is_compiled():
    """
    Verify that the model is compiled with the expected
    loss function and optimizer.
    """

    model = build_test_model()

    assert model.optimizer is not None
    assert model.loss == (
        "categorical_crossentropy"
    )


def test_prediction_shape():
    """
    Verify that predictions have the expected dimensions.
    """

    model = build_test_model()

    test_images = np.random.random(
        (2, 224, 224, 3)
    ).astype(np.float32)

    predictions = model.predict(
        test_images,
        verbose=0
    )

    assert predictions.shape == (
        2,
        NUM_CLASSES
    )


def test_predictions_form_probabilities():
    """
    Verify that softmax predictions form valid probability
    distributions.
    """

    model = build_test_model()

    test_images = np.random.random(
        (2, 224, 224, 3)
    ).astype(np.float32)

    predictions = model.predict(
        test_images,
        verbose=0
    )

    # Every probability should be between 0 and 1
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)

    # Probabilities for each sample should sum to ~1
    assert np.allclose(
        predictions.sum(axis=1),
        1.0,
        atol=1e-5
    )


def test_model_has_trainable_layers():
    """
    Verify that the model contains trainable parameters
    after the transfer-learning configuration.
    """

    model = build_test_model()

    trainable_parameters = sum(
        np.prod(variable.shape)
        for variable in model.trainable_variables
    )

    assert trainable_parameters > 0


def test_model_accepts_expected_input():
    """
    Verify that the model accepts the image dimensions
    expected by ResNet50.
    """

    model = build_test_model()

    input_shape = model.input_shape

    assert input_shape == (
        None,
        224,
        224,
        3
    )
