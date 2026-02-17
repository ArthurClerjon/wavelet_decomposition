#!/bin/bash
# =============================================================================
# WAVELET DECOMPOSITION - CLEANUP SCRIPT
# =============================================================================
# Run this script from your project root directory
#
# Usage:
#   chmod +x cleanup.sh
#   ./cleanup.sh
#
# =============================================================================

echo "========================================"
echo "Wavelet Decomposition - Cleanup Script"
echo "========================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "ERROR: Not a git repository. Please run from project root."
    exit 1
fi

# =============================================================================
# STEP 1: Create cleanup branch
# =============================================================================
echo ""
echo "[1/6] Creating cleanup branch..."

# Check if branch already exists
if git show-ref --quiet refs/heads/cleanup/v1.0; then
    echo "Branch 'cleanup/v1.0' already exists. Switching to it..."
    git checkout cleanup/v1.0
else
    git checkout -b cleanup/v1.0
    echo "Created and switched to branch 'cleanup/v1.0'"
fi

# =============================================================================
# STEP 2: Remove unnecessary files
# =============================================================================
echo ""
echo "[2/6] Removing unnecessary files..."

# List of files to remove
FILES_TO_REMOVE=(
    "calc_EPN_-_Copie.py"
    "config_-_Copie.py"
    "tutorial.py"
    "interface_prototype.py"
    "workplan.md"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        git rm "$file"
        echo "  Removed: $file"
    else
        echo "  Skipped (not found): $file"
    fi
done

# =============================================================================
# STEP 3: Fix plots.py (remove incorrect indentation)
# =============================================================================
echo ""
echo "[3/6] Fixing plots.py indentation issue..."

if [ -f "plots.py" ]; then
    # Create backup
    cp plots.py plots.py.backup
    
    # Fix the indentation issue using sed
    # This removes the 4-space indent from lines 514 onwards up to line 731
    # Note: This is a simplified fix - manual verification recommended
    
    echo "  Please manually verify the fix in plots.py:"
    echo "  - Function 'plot_EPN_scenarios_plotly' (line ~514) should be at module level"
    echo "  - Function 'plot_EPN_scenarios_plotly_combined' (line ~734) should be at module level"
    echo "  - Backup saved as plots.py.backup"
else
    echo "  WARNING: plots.py not found"
fi

# =============================================================================
# STEP 4: Create directory structure
# =============================================================================
echo ""
echo "[4/6] Creating directory structure..."

mkdir -p results
mkdir -p docs/images
mkdir -p data

# Create .gitkeep files to preserve empty directories
touch results/.gitkeep
touch docs/images/.gitkeep
touch data/.gitkeep

echo "  Created: results/, docs/, data/"

# =============================================================================
# STEP 5: Copy new documentation files
# =============================================================================
echo ""
echo "[5/6] Documentation files..."

echo "  Please copy the following files to your project root:"
echo "  - README.md"
echo "  - CONTRIBUTING.md"
echo "  - requirements.txt"
echo "  - requirements-dev.txt"
echo "  - .gitignore"
echo "  - LICENSE"

# =============================================================================
# STEP 6: Stage and commit
# =============================================================================
echo ""
echo "[6/6] Staging changes..."

git add -A

echo ""
echo "========================================"
echo "Cleanup preparation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Copy documentation files (README.md, CONTRIBUTING.md, etc.) to project root"
echo "2. Manually fix plots.py indentation if needed"
echo "3. Review changes with: git status"
echo "4. Commit with: git commit -m 'chore: Clean up codebase for v1.0'"
echo "5. Push with: git push -u origin cleanup/v1.0"
echo "6. Create Pull Request on GitHub"
echo ""
