# 🌊 Wavelet Decomposition Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive tool for analyzing time series data using **wavelet decomposition**, based on the methodology from [Clerjon & Perdu (2019)](https://doi.org/10.1039/C8EE02660A).

![Wavelet Analysis Interface](docs/images/interface_preview.png)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## ✨ Features

- **📁 Excel Data Import**: Load time series from Excel files with automatic interpolation
- **🔬 Wavelet Decomposition**: 15 time scales from 45 minutes to 1 year
- **📊 Interactive Visualizations**:
  - Heatmaps of wavelet coefficients
  - FFT spectrum analysis
  - Signal reconstruction with selected scales
- **⚡ EPN Analysis**: Energy-Power-Number metrics for storage flexibility assessment
- **📥 HTML Export**: Generate reports with interactive Plotly charts

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/wavelet_decomposition.git
cd wavelet_decomposition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Streamlit Application

```bash
# Run the interactive web interface
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501`

### Jupyter Notebook

```bash
# Launch Jupyter and open the notebook
jupyter notebook interface_prototype.ipynb
```

## 📖 Usage

### Workflow Overview

```
1. Upload Data    → Excel file with Consumption, Wind, PV columns
2. Configure      → Select signal, year, region
3. Decompose      → Run wavelet decomposition (15 time scales)
4. Analyze        → View heatmaps, FFT, reconstructions
5. EPN Analysis   → Evaluate storage requirements for energy mixes
6. Export         → Download HTML report
```

### Data Format

Your Excel file should have:
- **Sheets**: One sheet per year (e.g., "2020", "2021", "2022")
- **Columns**: `Consumption`, `Wind`, `PV` (or similar)
- **Rows**: Time series data (e.g., 8760 hourly values per year)

Example structure:
```
Sheet "2020":
| Datetime | Consumption | Wind | PV |
|----------|-------------|------|-----|
| 2020-01-01 00:00 | 54000 | 12000 | 0 |
| 2020-01-01 01:00 | 52000 | 11500 | 0 |
| ... | ... | ... | ... |
```

### Time Scales

The decomposition uses **15 fixed time scales**:

| Category | Scales (hours) | Physical Meaning |
|----------|----------------|------------------|
| **Daily** | 0.75, 1.5, 3, 6, 12, 24 | Intra-day variations, demand peaks |
| **Weekly** | 42, 84, 168 | Workday/weekend patterns |
| **Yearly** | 273.75, 547.5, 1095, 2190, 4380, 8760 | Seasonal variations |

## 🔬 Methodology

This implementation follows the methodology described in:

> **A. Clerjon and F. Perdu**, "Matching intermittency and electricity storage characteristics through time scale analysis: an energy return on investment comparison", *Energy Environ. Sci.*, 2019, 12, 693-705.
> 
> DOI: [10.1039/C8EE02660A](https://doi.org/10.1039/C8EE02660A)

### Key Concepts

1. **Wavelet Decomposition**: Decomposes a signal into components at different time scales using square or sine wavelets.

2. **Translation Optimization**: Finds optimal circular shifts to maximize correlation between wavelets and signal.

3. **EPN Analysis**: Calculates Energy (E), Power (P), and Number of cycles (N) for storage flexibility assessment.

### Mathematical Foundation

The signal `S(t)` is decomposed as:

```
S(t) = Σᵢ βᵢ × Wᵢ(t - τᵢ)
```

Where:
- `βᵢ` = amplitude coefficient for scale `i`
- `Wᵢ` = wavelet function at scale `i`
- `τᵢ` = optimal translation for scale `i`

## 📁 Project Structure

```
wavelet_decomposition/
├── streamlit_app.py          # Main Streamlit interface
├── interface_prototype.ipynb # Jupyter notebook demo
│
├── wavelet_decomposition.py  # Core decomposition algorithms
├── calc_translations.py      # Translation optimization
├── calc_EPN.py               # EPN calculations
├── plots.py                  # Visualization functions
├── file_manager.py           # File path management
├── import_excel.py           # Data import utilities
├── config.py                 # Configuration settings
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
│
├── results/                  # Output directory (auto-created)
│   ├── {region}/
│   │   ├── {shape}/
│   │   │   ├── translations/
│   │   │   ├── matrices/
│   │   │   └── betas/
│
└── docs/                     # Documentation
    └── images/
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black --check .
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{clerjon2019matching,
  title={Matching intermittency and electricity storage characteristics through time scale analysis: an energy return on investment comparison},
  author={Clerjon, Arthur and Perdu, Fabien},
  journal={Energy \& Environmental Science},
  volume={12},
  number={2},
  pages={693--705},
  year={2019},
  publisher={Royal Society of Chemistry},
  doi={10.1039/C8EE02660A}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original methodology by Arthur Clerjon and Fabien Perdu
- Built with [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), and [NumPy](https://numpy.org/)

---

**Questions?** Open an [issue](https://github.com/YOUR_USERNAME/wavelet_decomposition/issues) or contact the maintainers.
