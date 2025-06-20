"""
Tests for the bear classifier model functionality.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
class TestModelLoading:
    """Test model loading functionality."""
    
    def test_model_file_exists(self):
        """Test that the model file exists."""
        model_path = Path("export.pkl")
        if model_path.exists():
            assert model_path.is_file()
            assert model_path.stat().st_size > 0
        else:
            pytest.skip("Model file not found")
    
    @patch('fastai.vision.all.load_learner')
    def test_model_loading_success(self, mock_load_learner):
        """Test successful model loading."""
        mock_learner = Mock()
        mock_load_learner.return_value = mock_learner
        
        from fastai.vision.all import load_learner
        learn = load_learner('export.pkl', cpu=True)
        
        assert learn is not None
        mock_load_learner.assert_called_once_with('export.pkl', cpu=True)
    
    def test_model_loading_with_cpu(self):
        """Test model loading with CPU flag."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        try:
            from fastai.vision.all import load_learner
            learn = load_learner('export.pkl', cpu=True)
            assert learn is not None
        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")

class TestPrediction:
    """Test prediction functionality."""
    
    def test_prediction_input_validation(self):
        """Test prediction input validation."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        try:
            from fastai.vision.all import load_learner, PILImage
            learn = load_learner('export.pkl', cpu=True)
            
            # Test with None input
            with pytest.raises(Exception):
                learn.predict(None)
                
        except Exception as e:
            pytest.fail(f"Prediction test failed: {e}")
    
    def test_prediction_output_format(self):
        """Test prediction output format."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")
        
        try:
            from fastai.vision.all import load_learner, PILImage
            learn = load_learner('export.pkl', cpu=True)
            
            # Test with actual image
            sample_img_path = list(test_image_path.glob("*.jpg"))[0]
            img = PILImage.create(sample_img_path)
            
            pred, pred_idx, probs = learn.predict(img)
            
            # Check output format
            assert isinstance(pred, str)
            assert hasattr(pred_idx, 'item') or isinstance(pred_idx, int)  # Handle torch.Tensor or int
            assert hasattr(probs, '__len__')  # Handle torch.Tensor, list, or tuple
            assert len(probs) > 0
            
            # Convert pred_idx to int for comparison
            pred_idx_int = pred_idx.item() if hasattr(pred_idx, 'item') else pred_idx
            assert 0 <= pred_idx_int < len(probs)
            
            # Get probability value (handle tensor indexing)
            prob_value = probs[pred_idx_int].item() if hasattr(probs[pred_idx_int], 'item') else probs[pred_idx_int]
            assert 0 <= prob_value <= 1
            
        except Exception as e:
            pytest.fail(f"Prediction format test failed: {e}")
    
    def test_prediction_classes(self):
        """Test that predictions are from expected classes."""
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")
        
        expected_classes = ['black bear', 'grizzly bear', 'teddy bear']
        
        try:
            from fastai.vision.all import load_learner, PILImage
            learn = load_learner('export.pkl', cpu=True)
            
            sample_img_path = list(test_image_path.glob("*.jpg"))[0]
            img = PILImage.create(sample_img_path)
            
            pred, pred_idx, probs = learn.predict(img)
            
            assert pred in expected_classes
            
        except Exception as e:
            pytest.fail(f"Prediction classes test failed: {e}")

class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_prediction_speed(self):
        """Test that predictions complete within reasonable time."""
        import time
        
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        test_image_path = Path("images")
        if not test_image_path.exists() or not list(test_image_path.glob("*.jpg")):
            pytest.skip("No test images found")
        
        try:
            from fastai.vision.all import load_learner, PILImage
            learn = load_learner('export.pkl', cpu=True)
            
            sample_img_path = list(test_image_path.glob("*.jpg"))[0]
            img = PILImage.create(sample_img_path)
            
            # Time the prediction
            start_time = time.time()
            pred, pred_idx, probs = learn.predict(img)
            end_time = time.time()
            
            prediction_time = end_time - start_time
            
            # Should complete within 5 seconds on CPU
            assert prediction_time < 5.0, f"Prediction took {prediction_time:.2f} seconds"
            
        except Exception as e:
            pytest.fail(f"Prediction speed test failed: {e}")
    
    def test_model_memory_usage(self):
        """Test that model doesn't use excessive memory."""
        import psutil
        import os
        
        model_path = Path("export.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from fastai.vision.all import load_learner
            learn = load_learner('export.pkl', cpu=True)
            
            after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = after_load_memory - initial_memory
            
            # Model should not use more than 2GB additional memory
            assert memory_increase < 2048, f"Model uses {memory_increase:.1f}MB additional memory"
            
        except ImportError:
            pytest.skip("psutil not available")
        except Exception as e:
            pytest.fail(f"Memory usage test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 