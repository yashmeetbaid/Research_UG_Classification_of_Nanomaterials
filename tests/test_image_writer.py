import os
import cv2
import numpy as np


def writeimages(image_list, out_dir):
    """
    Copy images into the output directory while
    preserving their class-folder structure.

    This mirrors the project's current implementation.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filepath_old in image_list:

        image = cv2.imread(filepath_old)

        if image is None:
            raise ValueError(
                f"Unable to read image: {filepath_old}"
            )

        filepath_parts = filepath_old.replace(
            "\\", "/"
        ).split("/")

        class_name = filepath_parts[-2]

        filepath_new = os.path.join(
            out_dir,
            class_name
        )

        if not os.path.exists(filepath_new):
            os.makedirs(filepath_new)

        filepath_new = os.path.join(
            filepath_new,
            filepath_parts[-1]
        )

        cv2.imwrite(
            filepath_new,
            image
        )


def create_test_images(tmp_path):
    """
    Create test images using the same directory
    structure as the real dataset.
    """

    source_dir = tmp_path / "source"

    class_a = source_dir / "class_a"
    class_b = source_dir / "class_b"

    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    image_paths = []

    for class_dir in [class_a, class_b]:

        for i in range(2):

            image = np.zeros(
                (100, 100, 3),
                dtype=np.uint8
            )

            image[:] = i * 100

            image_path = class_dir / f"image_{i}.jpg"

            cv2.imwrite(
                str(image_path),
                image
            )

            image_paths.append(str(image_path))

    return image_paths


def test_output_directory_is_created(tmp_path):
    """
    Verify that writeimages creates the output
    directory when it does not exist.
    """

    image_paths = create_test_images(tmp_path)

    output_dir = tmp_path / "training"

    assert not output_dir.exists()

    writeimages(
        image_paths,
        str(output_dir)
    )

    assert output_dir.exists()


def test_class_directories_are_preserved(tmp_path):
    """
    Verify that the original class-folder structure
    is preserved in the output dataset.
    """

    image_paths = create_test_images(tmp_path)

    output_dir = tmp_path / "training"

    writeimages(
        image_paths,
        str(output_dir)
    )

    assert (output_dir / "class_a").exists()
    assert (output_dir / "class_b").exists()


def test_images_are_written(tmp_path):
    """
    Verify that every input image is written to the
    expected output directory.
    """

    image_paths = create_test_images(tmp_path)

    output_dir = tmp_path / "training"

    writeimages(
        image_paths,
        str(output_dir)
    )

    for image_path in image_paths:

        filename = os.path.basename(image_path)

        path_parts = image_path.replace(
            "\\", "/"
        ).split("/")

        class_name = path_parts[-2]

        output_path = (
            output_dir /
            class_name /
            filename
        )

        assert output_path.exists()


def test_written_images_are_readable(tmp_path):
    """
    Verify that the generated image files are valid
    and can be read by OpenCV.
    """

    image_paths = create_test_images(tmp_path)

    output_dir = tmp_path / "training"

    writeimages(
        image_paths,
        str(output_dir)
    )

    for image_path in image_paths:

        filename = os.path.basename(image_path)

        path_parts = image_path.replace(
            "\\", "/"
        ).split("/")

        class_name = path_parts[-2]

        output_path = (
            output_dir /
            class_name /
            filename
        )

        image = cv2.imread(
            str(output_path)
        )

        assert image is not None
        assert image.size > 0
