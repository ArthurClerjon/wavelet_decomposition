# Cleanup Package Contents

This package contains all files needed to clean up the wavelet decomposition project.

## Files Included

| File | Copy To | Purpose |
|------|---------|---------|
| `README.md` | Project root | Main documentation |
| `CONTRIBUTING.md` | Project root | Co-development guidelines |
| `requirements.txt` | Project root | Python dependencies |
| `requirements-dev.txt` | Project root | Dev dependencies |
| `LICENSE` | Project root | MIT License |
| `.gitignore` | Project root | Git ignore rules |
| `cleanup.sh` | Project root | Automated cleanup script |
| `fix_plots_indentation.py` | Project root | Fix plots.py bug |
| `CLEANUP_INSTRUCTIONS.md` | (reference only) | Step-by-step guide |

## Quick Start

```bash
# 1. Copy all files to your project
cp README.md CONTRIBUTING.md requirements.txt requirements-dev.txt LICENSE .gitignore ~/your-project/

# 2. Copy scripts
cp cleanup.sh fix_plots_indentation.py ~/your-project/

# 3. Run cleanup
cd ~/your-project
chmod +x cleanup.sh
./cleanup.sh

# 4. Fix plots.py
python fix_plots_indentation.py

# 5. Commit and push
git add -A
git commit -m "chore: Clean up codebase for v1.0"
git push -u origin cleanup/v1.0
```

## After Cleanup

Your project structure should be:

```
wavelet_decomposition/
├── README.md                 ← NEW
├── CONTRIBUTING.md           ← NEW
├── requirements.txt          ← NEW
├── requirements-dev.txt      ← NEW
├── LICENSE                   ← NEW
├── .gitignore                ← UPDATED
│
├── streamlit_app.py          ✓ Keep
├── interface_prototype.ipynb ✓ Keep
├── wavelet_decomposition.py  ✓ Keep
├── calc_translations.py      ✓ Keep
├── calc_EPN.py               ✓ Keep
├── plots.py                  ✓ Keep (FIXED)
├── file_manager.py           ✓ Keep
├── import_excel.py           ✓ Keep
├── config.py                 ✓ Keep
│
├── results/                  ← NEW directory
├── docs/                     ← NEW directory
└── data/                     ← NEW directory
```

## Removed Files

- `calc_EPN_-_Copie.py` (backup)
- `config_-_Copie.py` (backup)
- `tutorial.py` (old)
- `interface_prototype.py` (generated from notebook)
- `workplan.md` (internal)
