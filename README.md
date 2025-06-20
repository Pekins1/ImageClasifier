# Bear Classifier

A machine learning project that classifies different types of bears using computer vision and deep learning techniques.

## Project Overview

This project implements a bear classifier that can distinguish between:

- Black bears
- Grizzly bears
- Teddy bears

## Features

- Image classification using deep learning
- Jupyter notebook for model development and testing
- Pre-trained model export functionality
- Test suite for validation

## Project Structure

```
project/
├── bears/              # Training data - black bear, grizzly bear, teddy bear
├── bears1/             # Additional training data
├── images/             # Test images
├── tests/              # Test files
├── bearClassifier.ipynb # Main classification notebook
├── Jupyter_Notebook.ipynb # Additional notebook
├── requirements.txt    # Python dependencies
└── export.pkl         # Exported model
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Activate virtual environment (if using):
   ```bash
   source venv/bin/activate  # On Unix/Mac
   venv\Scripts\activate     # On Windows
   ```

## Usage

1. Open `bearClassifier.ipynb` in Jupyter Notebook
2. Run the cells to train and test the model
3. Use the exported model for predictions

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

[Add your license here]
