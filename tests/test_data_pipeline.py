import os
import cv2
import numpy as np
import pytest

from sklearn.model_selection import train_test_split


def create_test_dataset(base_dir):
    """
    Creates a temporary image dataset with the same
    class-folder structure expected by the project.

    Example:
        dataset/
        ├── class_a/
        │   ├── image_1.jpg
        │   ├── image_2.jpg
        │   └── image_3.jpg
        └── class_b/
            ├── image_1.jpg
            ├── image_2.jpg
            └── image_3.jpg
    """

    classes = ["class_a", "class_b"]

    for class_name in classes:
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True)

        for i in range(3):
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            # Give each image different pixel values
            image[:] = i * 50

            image_path = class_dir / f"image_{i}.jpg"

            success = cv2.imwrite(
                str(image_path),
                image
            )

            assert success


def test_images_are_discovered(tmp_path):
    """
    Verify that JPG images are correctly discovered
    from class-specific directories.
    """

    dataset_dir = tmp_path / "dataset"

    create_test_dataset(dataset_dir)

    image_list = list(dataset_dir.glob("*/*.jpg"))

    assert len(image_list) == 6

    for image_path in image_list:
        assert image_path.suffix.lower() == ".jpg"


def test_dataset_is_split_correctly(tmp_path):
    """
    Verify that the dataset can be split into training
    and validation subsets.
    """

    dataset_dir = tmp_path / "dataset"

    create_test_dataset(dataset_dir)

    image_list = list(dataset_dir.glob("*/*.jpg"))

    train_samples, validation_samples = train_test_split(
        image_list,
        test_size=0.2,
        random_state=42
    )

    assert len(train_samples) == 4
    assert len(validation_samples) == 2

    # No sample should appear in both sets
    assert set(train_samples).isdisjoint(
        set(validation_samples)
    )


def test_split_is_reproducible(tmp_path):
    """
    Verify that using random_state=42 produces
    the same split every time.
    """

    dataset_dir = tmp_path / "dataset"

    create_test_dataset(dataset_dir)

    image_list = list(dataset_dir.glob("*/*.jpg"))

    train_1, test_1 = train_test_split(
        image_list,
        test_size=0.2,
        random_state=42
    )

    train_2, test_2 = train_test_split(
        image_list,
        test_size=0.2,
        random_state=42
    )

    assert train_1 == train_2
    assert test_1 == test_2


def test_empty_dataset_is_detected(tmp_path):
    """
    Verify that an empty dataset is correctly detected.
    """

    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()

    image_list = list(dataset_dir.glob("*/*.jpg"))

    assert len(image_list) == 0
