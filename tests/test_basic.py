def test_imports():
    """Test that required packages can be imported."""
    import fastai
    import torch
    assert True  # If we get here, imports worked

def test_basic_math():
    """Test basic functionality."""
    assert 1 + 1 == 2 