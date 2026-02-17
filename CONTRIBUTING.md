# Contributing to Wavelet Decomposition Analysis

Thank you for your interest in contributing! This document provides guidelines and tips for co-developing this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Feature Suggestions](#feature-suggestions)
- [Architecture Overview](#architecture-overview)

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/wavelet_decomposition.git
cd wavelet_decomposition

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/wavelet_decomposition.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

### 3. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 🔄 Development Workflow

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-csv-export` |
| Bug Fix | `fix/description` | `fix/slider-coupling` |
| Documentation | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/split-plots-module` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(epn): add multi-year comparison"
git commit -m "fix(plots): correct heatmap color scaling"
git commit -m "docs: update installation instructions"
```

## 🎨 Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Good: Descriptive names, type hints
def calculate_epn(
    betas: List[np.ndarray],
    satisfactions: List[float],
    time_scales: List[float],
    load_factor: float
) -> Dict[str, np.ndarray]:
    """
    Calculate Energy-Power-Number metrics.
    
    Args:
        betas: Wavelet coefficients for each time scale
        satisfactions: Target satisfaction rates (0-100)
        time_scales: List of time scales in hours
        load_factor: Average load in MW
    
    Returns:
        Dictionary with 'emax', 'pmax', 'uf', 'n', 'serv' arrays
    """
    pass

# Bad: Cryptic names, no documentation
def calc(b, s, t, l):
    pass
```

### Streamlit Conventions

```python
# Use session state for persistence
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Use unique keys for widgets
st.slider("Value", key=f"slider_{unique_id}")

# Group related UI elements
with st.expander("Advanced Options"):
    option1 = st.checkbox("Option 1")
    option2 = st.slider("Option 2", 0, 100)
```

### File Organization

```python
# Module header
"""
Module description
"""

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from file_manager import WaveletFileManager
from config import RESULTS_DIR

# Constants
DEFAULT_TIME_SCALES = [0.75, 1.5, 3., 6., 12, 24., ...]

# Functions/Classes
def main_function():
    pass

# Main execution (if applicable)
if __name__ == "__main__":
    main()
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test file
pytest tests/test_wavelet_decomposition.py
```

### Writing Tests

```python
# tests/test_wavelet_decomposition.py
import pytest
import numpy as np
from wavelet_decomposition import reconstruct

class TestReconstruct:
    def test_reconstruct_returns_array(self):
        """Reconstruction should return numpy array."""
        result = reconstruct(
            time_scales=[24., 168.],
            reconstructed_time_scales=[24.],
            matrix=mock_matrix,
            beta_sheet=mock_betas,
            ...
        )
        assert isinstance(result, np.ndarray)
    
    def test_reconstruct_preserves_length(self):
        """Reconstructed signal should match input length."""
        result = reconstruct(...)
        assert len(result) == expected_length
```

## 📝 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] No merge conflicts with `main`

### 2. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How was this tested?

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
```

### 3. Review Process

1. Submit PR to `main` branch
2. Wait for CI checks to pass
3. Address reviewer feedback
4. Squash and merge when approved

## 💡 Feature Suggestions

### High Priority (Requested)

1. **Multi-year Analysis**
   - Compare decompositions across years
   - Trend analysis over time

2. **CSV/Parquet Import**
   - Support for non-Excel formats
   - Large file handling

3. **Batch Processing**
   - Process multiple signals automatically
   - Command-line interface

4. **API/Web Service**
   - REST API for decomposition
   - Integration with other tools

### Medium Priority

5. **Additional Wavelet Types**
   - Morlet wavelets
   - Custom wavelet definitions

6. **Improved EPN Visualization**
   - Comparison charts
   - Sensitivity analysis plots

7. **Report Templates**
   - PDF export
   - Customizable templates

8. **Data Validation**
   - Input quality checks
   - Missing data handling

### Low Priority / Future

9. **Machine Learning Integration**
   - Anomaly detection
   - Pattern recognition

10. **Real-time Analysis**
    - Streaming data support
    - Live dashboard

## 🏗️ Architecture Overview

### Module Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                        streamlit_app.py                         │
│                    (User Interface Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   plots.py   │  │  calc_EPN.py │  │import_excel.py│          │
│  │(Visualization)│  │  (Analysis)  │  │   (I/O)      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│  ┌──────┴─────────────────┴─────────────────┴───────┐          │
│  │           wavelet_decomposition.py               │          │
│  │              (Core Algorithms)                   │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────┐          │
│  │              file_manager.py                     │          │
│  │            (File Path Management)                │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Excel File
    │
    ▼
import_excel() ──────────────────────────────┐
    │                                        │
    ▼                                        │
stacked_data (normalized arrays)             │
    │                                        │
    ▼                                        │
wavelet_decomposition_single_TS()            │
    │                                        │
    ├──► translations (cached)               │
    ├──► matrix (cached)                     │
    └──► betas (coefficients)                │
            │                                │
            ▼                                │
    ┌───────┴───────┐                        │
    │               │                        │
    ▼               ▼                        │
reconstruct()   calc_epn()                   │
    │               │                        │
    ▼               ▼                        │
plots.py ◄──────────┘◄───────────────────────┘
    │
    ▼
Visualizations (Matplotlib/Plotly)
```

## 🔧 Tips for Co-Development

### 1. Communication

- Use GitHub Issues for bugs and features
- Use Discussions for questions
- Tag relevant people in PRs

### 2. Modular Development

Each contributor can focus on specific modules:

| Module | Focus Area |
|--------|------------|
| `streamlit_app.py` | UI/UX improvements |
| `plots.py` | New visualizations |
| `wavelet_decomposition.py` | Algorithm improvements |
| `calc_EPN.py` | Analysis metrics |
| `file_manager.py` | Storage/caching |

### 3. Avoiding Conflicts

- Work on separate features
- Keep PRs small and focused
- Sync frequently with `main`

```bash
# Before starting work
git fetch upstream
git rebase upstream/main
```

### 4. Documentation First

For new features:
1. Write docstring first
2. Create example usage
3. Then implement

### 5. Use Feature Flags

For experimental features:

```python
EXPERIMENTAL_FEATURES = {
    'multi_year_comparison': False,
    'csv_import': True,
}

if EXPERIMENTAL_FEATURES['csv_import']:
    # New code
    pass
```

---

## Questions?

- 📧 Contact maintainers
- 💬 Open a Discussion
- 🐛 File an Issue

Thank you for contributing! 🙏
