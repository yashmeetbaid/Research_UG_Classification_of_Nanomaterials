import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_test_dataset():
    """
    Generate a small deterministic dataset for unit testing.

    This dataset is only used to verify that the ML pipeline
    behaves correctly.
    """

    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    return X, y


def train_model(X_train, y_train):
    """Train the classification model."""

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def test_dataset_has_expected_shape():
    """Verify that the generated test dataset has valid dimensions."""

    X, y = create_test_dataset()

    assert X.shape[0] == 200
    assert X.shape[1] == 10
    assert len(y) == 200


def test_train_test_split():
    """Verify that the train/test split behaves correctly."""

    X, y = create_test_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40


def test_model_can_be_trained():
    """Verify that the model can successfully learn from the data."""

    X, y = create_test_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_prediction_shape():
    """Verify that predictions match the number of test samples."""

    X, y = create_test_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_predictions_are_valid_classes():
    """Verify that predictions contain only known class labels."""

    X, y = create_test_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    unique_classes = set(np.unique(y))

    assert set(np.unique(predictions)).issubset(unique_classes)


def test_model_accuracy_is_reasonable():
    """
    Basic regression test to ensure the model is actually
    learning rather than producing random predictions.
    """

    X, y = create_test_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.70
