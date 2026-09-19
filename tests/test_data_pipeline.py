import os
import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


def create_test_dataset(base_dir):
    """
    Create a small temporary image dataset with the same
    directory structure expected by flow_from_directory().
    """

    classes = ["class_a", "class_b"]

    for split in ["train", "val", "test"]:

        for class_name in classes:

            class_dir = base_dir / split / class_name
            class_dir.mkdir(parents=True)

            for i in range(2):

                image = np.random.randint(
                    0,
                    256,
                    size=(224, 224, 3),
                    dtype=np.uint8
                )

                image_path = class_dir / f"image_{i}.jpg"

                tf.keras.utils.save_img(
                    str(image_path),
                    image
                )


def create_generator(directory):
    """
    Create the same type of generator used by the project.
    """

    datagen = ImageDataGenerator(
        preprocessing_function=
        tf.keras.applications.resnet50.preprocess_input
    )

    return datagen.flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )


def test_dataset_directories_exist(tmp_path):
    """
    Verify that train, validation and test directories
    can be created and detected.
    """

    create_test_dataset(tmp_path)

    assert (tmp_path / "train").exists()
    assert (tmp_path / "val").exists()
    assert (tmp_path / "test").exists()


def test_class_directories_exist(tmp_path):
    """
    Verify that each dataset split contains the expected
    class-specific directories.
    """

    create_test_dataset(tmp_path)

    for split in ["train", "val", "test"]:

        assert (
            tmp_path / split / "class_a"
        ).exists()

        assert (
            tmp_path / split / "class_b"
        ).exists()


def test_generator_detects_correct_number_of_classes(tmp_path):
    """
    Verify that flow_from_directory correctly identifies
    the number of classes.
    """

    create_test_dataset(tmp_path)

    generator = create_generator(
        str(tmp_path / "train")
    )

    assert generator.num_classes == 2


def test_generator_detects_images(tmp_path):
    """
    Verify that images are correctly loaded by the
    TensorFlow data generator.
    """

    create_test_dataset(tmp_path)

    generator = create_generator(
        str(tmp_path / "train")
    )

    assert generator.samples == 4


def test_image_batch_shape(tmp_path):
    """
    Verify that generated image batches have the expected
    ResNet50 input dimensions.
    """

    create_test_dataset(tmp_path)

    generator = create_generator(
        str(tmp_path / "train")
    )

    images, labels = next(generator)

    assert images.shape[1:] == (
        224,
        224,
        3
    )


def test_label_batch_shape(tmp_path):
    """
    Verify that categorical labels have the correct
    number of classes.
    """

    create_test_dataset(tmp_path)

    generator = create_generator(
        str(tmp_path / "train")
    )

    images, labels = next(generator)

    assert labels.shape[1] == 2


def test_test_generator_is_not_shuffled(tmp_path):
    """
    The test generator should preserve ordering so that
    predictions can be correctly compared with true labels.
    """

    create_test_dataset(tmp_path)

    generator = create_generator(
        str(tmp_path / "test")
    )

    assert generator.shuffle is False
