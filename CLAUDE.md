# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an SLA Predictor system - a machine learning-based tool that predicts whether a daily feed will meet or miss its Service Level Agreement (SLA) on a given date. It's a binary classification problem using historical SLA outcome data.

**Current Status**: Stage 1 completed (Data Preprocessing Foundation) with 96% test coverage and 29 passing tests.

## Architecture

The system follows a modular TDD approach with these core components:

- **Data Preprocessing** (`src/sla_predictor/data_loader.py`, `date_processor.py`, `sla_validator.py`): Handles CSV loading, date standardization, and SLA outcome validation
- **Feature Engineering** (Stage 2): Calendar/temporal features, holiday integration, lag variables
- **Model Training** (Stage 4): scikit-learn models (Logistic Regression, Decision Tree, Random Forest)
- **Prediction API** (Stage 5): CLI interface for predictions

## Development Commands

### Testing
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_loader.py -v

# Run with coverage
python -m pytest tests/ --cov=src/sla_predictor
```

### Python Environment
- Uses Python 3.12+ with virtual environment in `venv/`
- Always activate venv before running Python commands: `source venv/bin/activate`

## Development Approach

**Strict Test-Driven Development**: This project follows a 5-stage TDD plan. Each feature MUST be developed using RED-GREEN-REFACTOR cycles:
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor for quality and maintainability

**Stage Progress**:
- ✅ Stage 1: Data Preprocessing Foundation (COMPLETED - 96% coverage, 29 tests)
- 🚧 Stage 2: Feature Engineering Core (NEXT)
- ⏳ Stage 3: Advanced Features with Lag Variables
- ⏳ Stage 4: Model Training and Evaluation Pipeline
- ⏳ Stage 5: Prediction API and CLI Interface

## Code Standards

- **File Size Limit**: Maximum 800 lines per file (currently all files under 120 lines)
- **Function Size**: Maximum 80 lines per function
- **Test Coverage**: Target >95% coverage
- **Logging**: Comprehensive logging implemented using Python logging module
- **Error Handling**: Robust exception handling with clear error messages

## Key Dependencies

- **pandas**: Data manipulation
- **scikit-learn**: ML models (planned for Stage 4)
- **pytest**: Testing framework with coverage plugin
- **pathlib**: File path handling
- **logging**: Built-in Python logging

## File Structure

```
src/sla_predictor/          # Main package
├── data_loader.py          # CSV loading and validation (115 lines)
├── date_processor.py       # Date standardization (43 lines)
└── sla_validator.py        # SLA outcome validation (52 lines)

tests/                      # Test files mirror src structure
├── test_data_loader.py     # 8 comprehensive tests
├── test_date_processor.py  # 10 date processing tests
└── test_sla_validator.py   # 11 SLA validation tests

resources/
├── prd/sla_predictor.md    # Detailed system design document
└── development_plan/tdd_5_stage_plan.md  # TDD implementation plan
```

## Development Notes

- Each module has comprehensive docstrings for classes and methods
- Error messages are user-friendly and logged appropriately
- All file paths use pathlib.Path for cross-platform compatibility
- Tests use temporary files and proper cleanup
- Validation logic supports flexible input formats (0/1, Yes/No, True/False)