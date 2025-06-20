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
- Comprehensive CI/CD pipeline
- Code quality checks and automated testing

## Project Structure

```
project/
├── bears/              # Training data - black bear, grizzly bear, teddy bear
├── bears1/             # Additional training data
├── images/             # Test images
│   ├── test_basic.py   # Basic functionality tests
│   ├── test_model.py   # Model-specific tests
│   ├── test_data.py    # Data validation tests
│   └── test_integration.py # Integration tests
├── .github/workflows/  # GitHub Actions CI/CD
├── bearClassifier.ipynb # Main classification notebook
├── Jupyter_Notebook.ipynb # Additional notebook
├── requirements.txt    # Python dependencies
├── requirements-dev.txt # Development dependencies
├── pyproject.toml      # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
└── export.pkl         # Exported model
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For development, install additional dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Activate virtual environment (if using):
   ```bash
   source venv/bin/activate  # On Unix/Mac
   venv\Scripts\activate     # On Windows
   ```

## Usage

1. Open `bearClassifier.ipynb` in Jupyter Notebook
2. Run the cells to train and test the model
3. Use the exported model for predictions

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_model.py
pytest tests/test_data.py
pytest tests/test_integration.py

# Run tests in parallel
pytest -n auto
```

### Test Categories

- **Unit Tests**: Basic functionality and model operations
- **Data Tests**: Dataset validation and integrity checks
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Speed and memory usage validation

## CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline using GitHub Actions:

### Automated Checks

1. **Code Quality**:

   - Black code formatting
   - isort import sorting
   - flake8 linting
   - mypy type checking
   - bandit security scanning

2. **Testing**:

   - Unit tests with pytest
   - Coverage reporting
   - Data validation tests
   - Model functionality tests

3. **Security**:

   - Trivy vulnerability scanning
   - Dependency security checks

4. **Build & Deploy**:
   - Package building
   - Artifact uploads

### Pre-commit Hooks

Install pre-commit hooks for local development:

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

## Code Quality

### Formatting

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check formatting
black --check .
isort --check-only .
```

### Linting

```bash
# Run flake8
flake8 .

# Run mypy type checking
mypy .

# Run security scan
bandit -r .
```

## Requirements

### Production Dependencies

See `requirements.txt` for the complete list of dependencies.

### Development Dependencies

See `requirements-dev.txt` for development tools and testing frameworks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and code quality checks
5. Submit a pull request

### Development Workflow

1. Install development dependencies
2. Set up pre-commit hooks
3. Make changes in a feature branch
4. Run tests locally
5. Ensure all CI checks pass
6. Submit PR

## License

[Add your license here]
