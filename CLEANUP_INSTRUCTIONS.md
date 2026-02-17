# 🧹 Cleanup Instructions

Follow these steps to clean up the wavelet decomposition project.

## Prerequisites

- Git installed and configured
- Python 3.9+
- Access to your project repository

## Step-by-Step Instructions

### 1. Open Terminal in Your Project Directory

```bash
cd path/to/wavelet_decomposition
```

### 2. Create the Cleanup Branch

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create the cleanup branch
git checkout -b cleanup/v1.0
```

### 3. Remove Unnecessary Files

```bash
# Remove backup/duplicate files
git rm calc_EPN_-_Copie.py
git rm config_-_Copie.py

# Remove old files
git rm tutorial.py
git rm interface_prototype.py
git rm workplan.md
```

**Note:** Keep `interface_prototype.ipynb` (the Jupyter notebook) - only remove the `.py` version.

### 4. Fix plots.py Indentation Issue

**IMPORTANT:** There's a critical bug in `plots.py` where two functions are incorrectly nested inside another function.

#### Option A: Run the fix script (recommended)

```bash
# Copy fix_plots_indentation.py to your project
python fix_plots_indentation.py
```

#### Option B: Manual fix

1. Open `plots.py` in your editor
2. Find line ~514: `    def plot_EPN_scenarios_plotly(`
3. Remove the 4-space indentation from this line
4. Continue removing 4 spaces from all lines until line ~731 (`return figures`)
5. Do the same for `plot_EPN_scenarios_plotly_combined` (line ~734)

#### Verify the fix

```bash
# Should not produce any errors
python -c "from plots import plot_EPN_scenarios_plotly; print('OK')"
```

### 5. Copy Documentation Files

Copy these files to your project root:
- `README.md`
- `CONTRIBUTING.md`
- `requirements.txt`
- `requirements-dev.txt`
- `LICENSE`

Replace the existing `.gitignore` with the new version.

### 6. Create Directory Structure

```bash
mkdir -p results
mkdir -p docs/images
mkdir -p data

# Create .gitkeep files
touch results/.gitkeep
touch docs/images/.gitkeep
touch data/.gitkeep
```

### 7. Stage and Review Changes

```bash
# Stage all changes
git add -A

# Review what will be committed
git status
git diff --cached --stat
```

### 8. Commit the Cleanup

```bash
git commit -m "chore: Clean up codebase for v1.0

Removed:
- Backup files (*_-_Copie.py)
- Old tutorial.py
- Redundant interface_prototype.py
- Internal workplan.md

Fixed:
- plots.py indentation (plot_EPN_scenarios_plotly was nested)

Added:
- README.md with project documentation
- CONTRIBUTING.md for co-development guidelines
- requirements.txt and requirements-dev.txt
- LICENSE (MIT)
- Updated .gitignore
- Directory structure (results/, docs/, data/)"
```

### 9. Push to GitHub

```bash
git push -u origin cleanup/v1.0
```

### 10. Create Pull Request

1. Go to your GitHub repository
2. Click "Compare & pull request" for the `cleanup/v1.0` branch
3. Add description:
   ```
   ## Cleanup for v1.0 Release
   
   This PR cleans up the codebase for collaborative development:
   
   ### Removed
   - Backup/duplicate files
   - Old tutorial.py
   - Internal planning documents
   
   ### Fixed
   - Critical indentation bug in plots.py
   
   ### Added
   - Comprehensive README.md
   - CONTRIBUTING.md for co-development
   - requirements.txt
   - MIT License
   
   ### Ready for Review
   - [ ] All tests pass
   - [ ] Documentation is accurate
   - [ ] No sensitive data exposed
   ```
4. Request review if needed
5. Merge when approved

## Post-Cleanup: Share with Collaborators

### Invite Collaborators

1. Go to repository Settings → Collaborators
2. Add collaborators by GitHub username or email

### Create Issues for Future Work

Create GitHub Issues for planned features:

```
Title: Add CSV/Parquet import support
Labels: enhancement, good first issue
Description: Currently only Excel files are supported. Add support for CSV and Parquet formats.
```

```
Title: Add multi-year comparison visualization
Labels: enhancement
Description: Allow comparing wavelet decompositions across multiple years on the same plot.
```

```
Title: Add command-line interface
Labels: enhancement
Description: Create a CLI for batch processing without the Streamlit interface.
```

### Set Up Branch Protection (Optional)

1. Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable:
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date

## Files Summary

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main web interface |
| `interface_prototype.ipynb` | Jupyter notebook demo |
| `wavelet_decomposition.py` | Core algorithms |
| `calc_translations.py` | Translation optimization |
| `calc_EPN.py` | EPN calculations |
| `plots.py` | Visualization functions |
| `file_manager.py` | File management |
| `import_excel.py` | Data import |
| `config.py` | Configuration |
| `README.md` | Documentation |
| `CONTRIBUTING.md` | Contribution guide |
| `requirements.txt` | Dependencies |
| `LICENSE` | MIT License |

## Troubleshooting

### "Module not found" error after cleanup

Make sure you didn't accidentally delete a required file. Check imports in `streamlit_app.py`.

### plots.py syntax error

The indentation fix may not have worked correctly. Verify that `plot_EPN_scenarios_plotly` starts at column 0 (no leading spaces).

### Git conflicts

If someone else pushed changes:
```bash
git fetch origin
git rebase origin/main
# Resolve any conflicts
git push -f origin cleanup/v1.0
```

---

Questions? Open an issue on GitHub!
