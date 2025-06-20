"""
Integration tests for the bear classifier application.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEndToEndPipeline:
    """Test the complete end-to-end prediction pipeline."""

    def test_complete_prediction_workflow(self) -> None:
        """Test the complete workflow from image input to prediction output."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")

        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")

        try:
            from fastai.vision.all import PILImage, load_learner

            # Load model
            learn = load_learner("export.pkl", cpu=True)

            # Load test image
            sample_img_path = list(test_image_path.glob("*.jpg"))[0]
            img = PILImage.create(sample_img_path)

            # Make prediction
            pred, pred_idx, probs = learn.predict(img)

            # Validate outputs
            assert isinstance(pred, str)
            assert hasattr(pred_idx, "item") or isinstance(
                pred_idx, int
            )  # Handle torch.Tensor or int
            assert hasattr(probs, "__len__")  # Handle torch.Tensor, list, or tuple
            assert len(probs) > 0

            # Convert pred_idx to int for comparison
            pred_idx_int = pred_idx.item() if hasattr(pred_idx, "item") else pred_idx
            assert 0 <= pred_idx_int < len(probs)

            # Get probability value (handle tensor indexing)
            prob_value = (
                probs[pred_idx_int].item()
                if hasattr(probs[pred_idx_int], "item")
                else probs[pred_idx_int]
            )
            assert 0 <= prob_value <= 1

            # Check prediction is from expected classes
            expected_classes = ["black bear", "grizzly bear", "teddy bear"]
            assert pred in expected_classes

        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")

    def test_multiple_image_predictions(self) -> None:
        """Test predictions on multiple images."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")

        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")

        try:
            from fastai.vision.all import PILImage, load_learner

            learn = load_learner("export.pkl", cpu=True)
            image_files = list(test_image_path.glob("*.jpg"))

            predictions: list[str] = []
            for img_file in image_files[:3]:  # Test first 3 images
                img = PILImage.create(img_file)
                pred, pred_idx, probs = learn.predict(img)
                predictions.append(pred)

            # All predictions should be valid
            expected_classes = ["black bear", "grizzly bear", "teddy bear"]
            for pred in predictions:
                assert pred in expected_classes

        except Exception as e:
            pytest.fail(f"Multiple image test failed: {e}")


class TestFileUploadWorkflow:
    """Test file upload and processing workflow."""

    @patch("fastai.vision.all.PILImage")
    @patch("fastai.vision.all.load_learner")
    def test_file_upload_simulation(
        self, mock_load_learner: Mock, mock_pil_image: Mock
    ) -> None:
        """Simulate file upload workflow."""
        # Mock the learner
        mock_learner = Mock()
        mock_learner.predict.return_value = ("black bear", 0, [0.8, 0.1, 0.1])
        mock_load_learner.return_value = mock_learner

        # Mock the image
        mock_img = Mock()
        mock_pil_image.create.return_value = mock_img

        try:
            from fastai.vision.all import PILImage, load_learner

            # Simulate the workflow
            learn = load_learner("export.pkl", cpu=True)

            # Simulate file upload data (mock)
            upload_data = b"fake_image_data"

            # Create image from upload data
            img = PILImage.create(upload_data)

            # Make prediction
            pred, pred_idx, probs = learn.predict(img)

            # Validate
            assert pred == "black bear"
            assert pred_idx == 0
            assert len(probs) == 3

        except Exception as e:
            pytest.fail(f"File upload simulation failed: {e}")


class TestErrorHandling:
    """Test error handling in the application."""

    def test_invalid_image_handling(self) -> None:
        """Test handling of invalid image files."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")

        try:
            from fastai.vision.all import load_learner

            learn = load_learner("export.pkl", cpu=True)

            # Test with None
            with pytest.raises(Exception):
                learn.predict(None)

            # Test with invalid data
            with pytest.raises(Exception):
                learn.predict("invalid_data")

        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")

    def test_missing_model_file(self) -> None:
        """Test behavior when model file is missing."""
        try:
            from fastai.vision.all import load_learner

            # This should raise an exception
            with pytest.raises(Exception):
                load_learner("nonexistent_model.pkl", cpu=True)

        except Exception:
            # This is expected behavior
            pass


class TestPerformanceIntegration:
    """Test performance characteristics in integration scenarios."""

    def test_concurrent_predictions(self) -> None:
        """Test handling multiple predictions in sequence."""
        import time

        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")

        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")

        try:
            from fastai.vision.all import PILImage, load_learner

            learn = load_learner("export.pkl", cpu=True)
            image_files = list(test_image_path.glob("*.jpg"))

            if len(image_files) < 2:
                pytest.skip("Need at least 2 test images")

            start_time = time.time()

            # Make multiple predictions
            for img_file in image_files[:3]:
                img = PILImage.create(img_file)
                learn.predict(img)

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete within reasonable time
            assert (
                total_time < 15.0
            ), f"Multiple predictions took too long: {total_time:.2f}s"

        except Exception as e:
            pytest.fail(f"Concurrent predictions test failed: {e}")

    def test_memory_usage_integration(self) -> None:
        """Test memory usage during integration workflow."""
        import os

        import psutil

        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")

        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")

        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            from fastai.vision.all import PILImage, load_learner

            # Load model
            learn = load_learner("export.pkl", cpu=True)

            # Make several predictions
            image_files = list(test_image_path.glob("*.jpg"))
            for img_file in image_files[:5]:
                img = PILImage.create(img_file)
                learn.predict(img)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable
            assert (
                memory_increase < 2048
            ), f"Memory usage too high: {memory_increase:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available")
        except Exception as e:
            pytest.fail(f"Memory usage integration test failed: {e}")


class TestDataPipeline:
    """Test data processing pipeline."""

    def test_image_preprocessing(self) -> None:
        """Test image preprocessing steps."""
        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")

        try:
            from fastai.vision.all import PILImage

            sample_img_path = list(test_image_path.glob("*.jpg"))[0]
            img = PILImage.create(sample_img_path)

            # Test image resizing
            resized_img = img.to_thumb(128, 128)
            assert resized_img.size[0] <= 128
            assert resized_img.size[1] <= 128

            # Test image format conversion
            assert hasattr(img, "to_thumb")

        except Exception as e:
            pytest.fail(f"Image preprocessing test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
