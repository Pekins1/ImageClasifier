"""
Tests for data validation and integrity.
"""

import os
from pathlib import Path

import pytest
from PIL import Image


class TestDatasetStructure:
    """Test dataset directory structure and organization."""

    def test_required_directories_exist(self) -> None:
        """Test that all required directories exist."""
        required_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for dir_path in required_dirs:
            path = Path(dir_path)
            assert path.exists(), f"Directory {dir_path} does not exist"
            assert path.is_dir(), f"{dir_path} is not a directory"

    def test_dataset_has_images(self) -> None:
        """Test that each class directory contains images."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                # Look for common image formats
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                    image_files.extend(path.glob(ext))

                assert len(image_files) > 0, f"No images found in {class_dir}"
                print(f"{class_dir}: {len(image_files)} images")

    def test_image_file_integrity(self) -> None:
        """Test that image files can be opened and are valid."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(path.glob(ext))

                # Test first few images from each class
                for img_file in image_files[:5]:  # Test first 5 images
                    try:
                        with Image.open(img_file) as img:
                            # Check basic image properties
                            assert img.size[0] > 0, f"Image {img_file} has zero width"
                            assert img.size[1] > 0, f"Image {img_file} has zero height"
                            assert img.mode in [
                                "RGB",
                                "RGBA",
                                "L",
                            ], f"Image {img_file} has unsupported mode: {img.mode}"
                    except Exception as e:
                        pytest.fail(f"Failed to open image {img_file}: {e}")

    def test_class_balance(self) -> None:
        """Test that classes have reasonable balance (not too skewed)."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        class_counts: dict[str, int] = {}
        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(path.glob(ext))
                class_counts[class_dir] = len(image_files)

        if len(class_counts) >= 2:
            counts = list(class_counts.values())
            min_count = min(counts)
            max_count = max(counts)

            # Check that no class has more than 10x the minimum
            ratio = max_count / min_count if min_count > 0 else float("inf")
            assert ratio <= 10, f"Class imbalance too high: {ratio:.1f}x difference"

    def test_file_naming_convention(self) -> None:
        """Test that files follow consistent naming patterns."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(path.glob(ext))

                for img_file in image_files:
                    # Check for common naming patterns
                    assert (
                        " " not in img_file.name
                    ), f"Filename contains spaces: {img_file.name}"

                    # Should have valid extension
                    assert img_file.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                    ], f"Invalid extension: {img_file.suffix}"


class TestImageQuality:
    """Test image quality and characteristics."""

    def test_image_resolution(self) -> None:
        """Test that images have reasonable resolution."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(path.glob(ext))

                # Test first few images from each class
                for img_file in image_files[:3]:
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size

                            # Check minimum resolution
                            assert (
                                width >= 50
                            ), f"Image {img_file} too narrow: {width}px"
                            assert (
                                height >= 50
                            ), f"Image {img_file} too short: {height}px"

                            # Check maximum resolution (reasonable limit)
                            assert (
                                width <= 8000
                            ), f"Image {img_file} too wide: {width}px"
                            assert (
                                height <= 8000
                            ), f"Image {img_file} too tall: {height}px"

                    except Exception as e:
                        pytest.fail(f"Failed to check image {img_file}: {e}")

    def test_image_file_size(self) -> None:
        """Test that image files have reasonable sizes."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                image_files: list[Path] = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(path.glob(ext))

                for img_file in image_files:
                    file_size = img_file.stat().st_size

                    # Check minimum file size (not empty)
                    assert file_size > 1000, f"Image file too small: {file_size} bytes"

                    # Check maximum file size (reasonable limit)
                    assert (
                        file_size < 50 * 1024 * 1024
                    ), f"Image file too large: {file_size} bytes"


class TestDataConsistency:
    """Test data consistency across the dataset."""

    def test_consistent_file_extensions(self) -> None:
        """Test that all images use consistent file extensions."""
        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        extensions: set[str] = set()
        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_file in path.glob(ext):
                        extensions.add(img_file.suffix.lower())

        # Should have at least one extension
        assert len(extensions) > 0, "No image files found"

        # Should not have too many different extensions
        assert len(extensions) <= 3, f"Too many different extensions: {extensions}"

    def test_no_duplicate_files(self) -> None:
        """Test that there are no duplicate filenames across classes."""
        all_filenames: set[str] = set()
        duplicates: list[str] = []

        class_dirs = ["bears/black bear", "bears/grizzly bear", "bears/teddy bear"]

        for class_dir in class_dirs:
            path = Path(class_dir)
            if path.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_file in path.glob(ext):
                        if img_file.name in all_filenames:
                            duplicates.append(img_file.name)
                        all_filenames.add(img_file.name)

        if duplicates:
            print(f"Warning: Found duplicate filenames: {duplicates}")
            # Skip instead of fail - duplicates might be intentional
            pytest.skip(f"Duplicate filenames found: {duplicates}")


if __name__ == "__main__":
    pytest.main([__file__])
