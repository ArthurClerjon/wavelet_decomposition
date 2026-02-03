import os

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base results directory
RESULTS_DIR = 'results'

# Subdirectories - dynamically constructed from RESULTS_DIR
TRANS_DIR = os.path.join(RESULTS_DIR, 'translation_calculation_results')
BETAS_DIR = os.path.join(RESULTS_DIR, 'beta_results')
MATRIX_DIR = os.path.join(RESULTS_DIR, 'saved_matrix', 'square_shape')

def get_matrix_dir(wl_shape='square'):
    """
    Get the matrix directory for a specific wavelet shape.
    
    Args:
        wl_shape: 'square' or 'sine'
        
    Returns:
        Full path to matrix directory for this shape
        
    Example:
        >>> get_matrix_dir('square')
        'results/saved_matrix/square_shape'
        >>> get_matrix_dir('sine')
        'results/saved_matrix/sine_shape'
    """
    return os.path.join(MATRIX_DIR_BASE, f'{wl_shape}_shape')

# ============================================================================
# DIRECTORY STRUCTURE REFERENCE
# ============================================================================
"""
After running decomposition, the directory structure will be:

results/
├── translation_calculation_results/    ← TRANS_DIR
│   └── trans_France_PV.pkl
├── saved_matrix/                       ← MATRIX_DIR_BASE
│   ├── square_shape/                   ← get_matrix_dir('square')
│   │   └── A_2012.npz
│   └── sine_shape/                     ← get_matrix_dir('sine')
│       └── A_2012.npz
└── beta_results/                       ← BETAS_DIR
    └── France/
        └── PV/
            ├── betas_France_PV_2012.pkl
            └── results_betas_stacked.xlsx
"""
# ============================================================================

#Todo: in the future, only define the parameters below hre and not in other files.

time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380.,
        8760.]
vy = 6  # number of vectors per years (child wavelets)
vw = 3  # vectors per week
vd = 6  # vectors per day
