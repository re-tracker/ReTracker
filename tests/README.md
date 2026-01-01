# ReTracker Tests

This directory contains comprehensive tests for the ReTracker project.

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_retracker.py        # Unit tests for ReTracker class
├── test_integration.py      # Integration tests
├── run_tests.py             # Test runner script
└── README.md               # This file
```

## Running Tests

### Using the test runner script
```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py -v

# Run specific test pattern
python tests/run_tests.py -p "test_config"

# List all available tests
python tests/run_tests.py --list
```

### Using pytest directly
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=retracker

# Run specific test file
pytest tests/test_retracker.py

# Run specific test class
pytest tests/test_retracker.py::TestReTracker

# Run specific test method
pytest tests/test_retracker.py::TestReTracker::test_config_constants

# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

### Using unittest
```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_retracker

# Run specific test class
python -m unittest tests.test_retracker.TestReTracker

# Run specific test method
python -m unittest tests.test_retracker.TestReTracker.test_config_constants
```

## Test Categories

### Unit Tests (`test_retracker.py`)
- **Configuration Tests**: Test ReTrackerConfig constants and validation
- **Data Preprocessing Tests**: Test data structure and preprocessing
- **Visibility Calculation Tests**: Test visibility and certainty calculations
- **Grid Query Generation Tests**: Test query generation and bounds checking
- **NaN Handling Tests**: Test handling of NaN values
- **Memory Management Tests**: Test memory storage and retrieval
- **Tensor Operations Tests**: Test basic tensor operations

### Integration Tests (`test_integration.py`)
- **End-to-End Matching Tests**: Test complete matching pipeline
- **Video Forward Tests**: Test video processing structure
- **Augmentation Tests**: Test augmentation logic
- **Error Handling Tests**: Test error scenarios
- **Performance Tests**: Test memory usage and operation timing

## Test Fixtures

The `conftest.py` file provides common fixtures:

- `device`: CPU/GPU device
- `basic_config`: Basic ReTracker configuration
- `mock_data`: Mock image and query data
- `mock_video_data`: Mock video data
- `retracker_instance`: ReTracker instance
- `small_tensor`: Small tensor for testing
- `coordinates`: Coordinate data
- `certainty_logits`: Certainty logits
- `occlusion_logits`: Occlusion logits
- `memory_dict`: Memory dictionary

## Test Markers

Tests are automatically marked based on their names:
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Performance tests (can be skipped)

## Adding New Tests

1. **Unit Tests**: Add to `test_retracker.py`
2. **Integration Tests**: Add to `test_integration.py`
3. **New Test Files**: Create new files with `test_` prefix
4. **Fixtures**: Add to `conftest.py` if shared

### Example Test Structure
```python
def test_new_feature(self):
    """Test description"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    self.assertIsNotNone(result)
    self.assertEqual(result.shape, expected_shape)
```

## Test Coverage

The tests cover:
- ✅ Configuration validation
- ✅ Data preprocessing
- ✅ Tensor operations
- ✅ Memory management
- ✅ Error handling
- ✅ Performance metrics
- ✅ Integration scenarios

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -e ".[test]"
    pytest tests/ --cov=retracker --cov-report=xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in Python path
2. **CUDA Errors**: Tests automatically fall back to CPU
3. **Memory Issues**: Tests use small tensors to avoid memory problems
4. **Model Loading**: Some tests skip if models aren't available

### Debug Mode
```bash
# Run with debug output
pytest tests/ -v -s

# Run specific failing test
pytest tests/test_retracker.py::TestReTracker::test_specific_method -v -s
``` 