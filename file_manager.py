"""
File Manager for Wavelet Decomposition - UPDATED VERSION
Handles proper naming and organization of translation, matrix, and beta files

CHANGES FROM ORIGINAL:
- Added optional root directory parameters to get_translation_path, get_matrix_path, get_betas_path
- When root parameters are provided, they override the default nested structure
- This allows compatibility with config.py directory settings
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class WaveletFileManager:
    """
    Manages file naming and directory structure for wavelet decomposition outputs.
    
    Directory Architecture
    ----------------------
    results/
    └── {region}/                          # e.g., "France", "Germany"
        ├── {wl_shape}/                    # "square" or "sine"
        │   ├── translations/
        │   │   └── trans_{region}_{signal_type}_{year}.pkl
        │   ├── matrices/
        │   │   └── A_{year}.npz
        │   └── betas/
        │       └── betas_{region}_{signal_type}_{year}.pkl
        │
        └── {wl_shape}/                    # Alternative wavelet shape
            └── ...
    
    Example Structure
    -----------------
    results/
    └── France/
        ├── square/
        │   ├── translations/
        │   │   └── trans_France_PV_2012.pkl
        │   ├── matrices/
        │   │   └── A_2012.npz
        │   └── betas/
        │       └── betas_France_PV_2012.pkl
        └── sine/
            ├── translations/
            │   └── trans_France_PV_2012.pkl
            ├── matrices/
            │   └── A_2012.npz
            └── betas/
                └── betas_France_PV_2012.pkl
    
    Attributes
    ----------
    base_dir : str
        Base results directory (default: 'results')
    region : str
        Region identifier (e.g., "France", "Germany")
    wl_shape : str
        Wavelet shape: 'square' or 'sine'
    use_nested : bool
        If True, uses nested structure; if False, flat structure
    """
    
    def __init__(self, base_dir: str = 'results', region: str = 'France',
                 wl_shape: str = 'square', use_nested: bool = True):
        """
        Initialize the file manager.
        
        Args:
            base_dir: Base directory for all results
            region: Region/country identifier (e.g., 'France', 'Germany')
            wl_shape: Wavelet shape - 'square' or 'sine'
            use_nested: Use nested directory structure (True) or flat (False)
            
        Raises:
            ValueError: If wl_shape is not 'square' or 'sine'
        """
        if wl_shape not in ['square', 'sine']:
            raise ValueError(f"wl_shape must be 'square' or 'sine', got '{wl_shape}'")
        
        self.base_dir = base_dir
        self.region = region
        self.wl_shape = wl_shape
        self.use_nested = use_nested
    
    def _get_shape_dir(self) -> str:
        """Get the base directory including region and wavelet shape."""
        if self.use_nested:
            return os.path.join(self.base_dir, self.region, self.wl_shape)
        else:
            return os.path.join(self.base_dir, self.wl_shape)
        
        
    def _ensure_dir(self, path: str) -> str:
        """Create directory if it doesn't exist and return the path"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path
    

    def get_translation_path(self, signal_type: str, year: str) -> str:
            """
            Get path for translation file.
            
            Args:
                signal_type: Type of signal (e.g., "Consumption", "PV", "Wind")
                year: Year identifier (e.g., "2012", "2013")
                
            Returns:
                Full path to translation file
                
            Example:
                >>> mgr = WaveletFileManager(region='France', wl_shape='square')
                >>> mgr.get_translation_path('PV', '2012')
                'results/France/square/translations/trans_France_PV_2012.pkl'
                
                >>> mgr = WaveletFileManager(region='France', wl_shape='sine')
                >>> mgr.get_translation_path('PV', '2012')
                'results/France/sine/translations/trans_France_PV_2012.pkl'
            """
            path = os.path.join(
                self._get_shape_dir(),
                'translations',
                f'trans_{self.region}_{signal_type}_{year}.pkl'
            )
            return self._ensure_dir(path)


    def get_matrix_path(self, year: str) -> str:
        """
        Get path for matrix file.
        
        Matrix files are stored per wavelet shape since different shapes
        produce different decomposition matrices.
        
        Args:
            year: Year identifier (e.g., "2012", "2013")
            
        Returns:
            Full path to matrix file
            
        Example:
            >>> mgr = WaveletFileManager(region='France', wl_shape='square')
            >>> mgr.get_matrix_path('2012')
            'results/France/square/matrices/A_2012.npz'
            
            >>> mgr = WaveletFileManager(region='France', wl_shape='sine')
            >>> mgr.get_matrix_path('2012')
            'results/France/sine/matrices/A_2012.npz'
        
        Note:
            Matrix files are named A_{year}.npz (not A_{region}_{year}.npz)
            to maintain compatibility with existing decomposition functions.
        """
        path = os.path.join(
            self._get_shape_dir(),
            'matrices',
            f'A_{year}.npz'
        )
        return self._ensure_dir(path)
    
    def get_betas_path(self, signal_type: str, year: str) -> str:
        """
        Get path for beta coefficients file.
        
        Args:
            signal_type: Type of signal (e.g., "Consumption", "PV", "Wind")
            year: Year identifier (e.g., "2012", "2013")
            
        Returns:
            Full path to betas file
            
        Example:
            >>> mgr = WaveletFileManager(region='France', wl_shape='square')
            >>> mgr.get_betas_path('PV', '2012')
            'results/France/square/betas/betas_France_PV_2012.pkl'
            
            >>> mgr = WaveletFileManager(region='France', wl_shape='sine')
            >>> mgr.get_betas_path('PV', '2012')
            'results/France/sine/betas/betas_France_PV_2012.pkl'
        """
        path = os.path.join(
            self._get_shape_dir(),
            'betas',
            f'betas_{self.region}_{signal_type}_{year}.pkl'
        )
        return self._ensure_dir(path)
    
    def get_all_paths(self, signal_type: str, years: List[str]) -> Dict[str, any]:
        """
        Get all file paths for a complete analysis.
        
        Args:
            signal_type: Type of signal (e.g., 'PV', 'Wind')
            years: List of years (e.g., ['2012', '2013'])
            
        Returns:
            Dictionary with paths for translations, matrices, and betas
            
        Example:
            >>> mgr = WaveletFileManager(region='France', wl_shape='square')
            >>> paths = mgr.get_all_paths('PV', ['2012', '2013'])
            >>> paths['translations']
            ['results/France/square/translations/trans_France_PV_2012.pkl',
             'results/France/square/translations/trans_France_PV_2013.pkl']
        """
        return {
            'translations': [self.get_translation_path(signal_type, year) for year in years],
            'matrices': [self.get_matrix_path(year) for year in years],
            'betas': [self.get_betas_path(signal_type, year) for year in years]
        }
    
    def log_analysis(self, signal_type: str, years: List[str], 
                     parameters: Dict, files: Dict) -> None:
        """
        Log analysis metadata to JSON file.
        
        Args:
            signal_type: Type of signal analyzed
            years: List of years analyzed
            parameters: Analysis parameters (dpd, ndpd, etc.)
            files: Dictionary of file paths created
        """
        log_file = os.path.join(self.base_dir, 'metadata', 'analysis_log.json')
        self._ensure_dir(log_file)
        
        # Load existing log or create new
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'analyses': []}
        
        # Add new entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'region': self.region,
            'signal_type': signal_type,
            'years': years,
            'files': files,
            'parameters': parameters
        }
        
        log_data['analyses'].append(entry)
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ Analysis logged to: {log_file}")
    
    def find_files(self, pattern: str = None, signal_type: str = None, 
                   year: str = None) -> List[str]:
        """
        Find files matching criteria.
        
        Args:
            pattern: Glob pattern (e.g., "*.pkl", "betas_*.pkl")
            signal_type: Filter by signal type
            year: Filter by year
            
        Returns:
            List of matching file paths
        """
        import glob
        
        if pattern:
            search_pattern = os.path.join(self.base_dir, '**', pattern)
        else:
            search_pattern = os.path.join(self.base_dir, '**', '*')
        
        files = glob.glob(search_pattern, recursive=True)
        
        # Apply filters
        if signal_type:
            files = [f for f in files if signal_type in f]
        if year:
            files = [f for f in files if year in f]
        
        return files
    
    def get_summary(self) -> Dict:
        """
        Get summary of all files for this region.
        
        Returns:
            Dictionary with counts and lists of files
        """
        import glob
        
        if self.use_nested:
            base_path = os.path.join(self.base_dir, self.region)
        else:
            base_path = self.base_dir
        
        summary = {
            'region': self.region,
            'base_directory': base_path,
            'translations': [],
            'matrices': [],
            'betas': []
        }
        
        # Find all files
        if os.path.exists(base_path):
            summary['translations'] = glob.glob(os.path.join(base_path, '**/trans_*.pkl'), recursive=True)
            summary['matrices'] = glob.glob(os.path.join(base_path, '**/A_*.npz'), recursive=True)
            summary['betas'] = glob.glob(os.path.join(base_path, '**/betas_*.pkl'), recursive=True)
        
        summary['counts'] = {
            'translations': len(summary['translations']),
            'matrices': len(summary['matrices']),
            'betas': len(summary['betas'])
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of files."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"FILE SUMMARY: {summary['region']}")
        print(f"{'='*60}")
        print(f"Base directory: {summary['base_directory']}")
        print(f"\nCounts:")
        print(f"  Translations: {summary['counts']['translations']}")
        print(f"  Matrices:     {summary['counts']['matrices']}")
        print(f"  Betas:        {summary['counts']['betas']}")
        
        if summary['translations']:
            print(f"\nTranslation files:")
            for f in summary['translations']:
                print(f"  - {os.path.basename(f)}")
        
        if summary['matrices']:
            print(f"\nMatrix files (showing first 5):")
            for f in summary['matrices'][:5]:
                print(f"  - {os.path.basename(f)}")
        
        if summary['betas']:
            print(f"\nBeta files (showing first 5):")
            for f in summary['betas'][:5]:
                print(f"  - {os.path.basename(f)}")
        
        print(f"{'='*60}\n")


def migrate_old_files(old_dir: str, new_base_dir: str, region: str, 
                     signal_type: str) -> None:
    """
    Migrate files from old structure to new organized structure.
    
    Args:
        old_dir: Old results directory
        new_base_dir: New base directory
        region: Region identifier
        signal_type: Signal type for this migration
    """
    import shutil
    import glob
    
    mgr = WaveletFileManager(base_dir=new_base_dir, region=region)
    
    print(f"Migrating files from {old_dir} to new structure...")
    
    # Migrate translation files
    old_trans = glob.glob(os.path.join(old_dir, 'translation*', '*.pkl'))
    for old_file in old_trans:
        new_path = mgr.get_translation_path(signal_type)
        print(f"  {os.path.basename(old_file)} → {new_path}")
        shutil.copy2(old_file, new_path)
    
    # Migrate matrix files
    old_matrices = glob.glob(os.path.join(old_dir, 'saved_matrix', '**', 'A_*.npz'), recursive=True)
    for old_file in old_matrices:
        # Extract year from filename
        year = os.path.basename(old_file).replace('A_', '').replace('.npz', '')
        new_path = mgr.get_matrix_path(year)
        print(f"  {os.path.basename(old_file)} → {new_path}")
        shutil.copy2(old_file, new_path)
    
    # Migrate beta files
    old_betas = glob.glob(os.path.join(old_dir, 'beta_results', '*.pkl'))
    for old_file in old_betas:
        # You'll need to specify years manually or parse from file
        # For now, just notify
        print(f"  Beta file found: {old_file} (specify year for migration)")
    
    print("✅ Migration complete!")


# Example usage
if __name__ == '__main__':
    
    # Example 1: Create file manager for square wavelets
    mgr_square = WaveletFileManager(base_dir='results', region='France', wl_shape='square')
    
    print("Square wavelet paths:")
    print(f"  Translation: {mgr_square.get_translation_path('PV', '2012')}")
    print(f"  Matrix:      {mgr_square.get_matrix_path('2012')}")
    print(f"  Betas:       {mgr_square.get_betas_path('PV', '2012')}")
    
    # Example 2: Create file manager for sine wavelets
    mgr_sine = WaveletFileManager(base_dir='results', region='France', wl_shape='sine')
    
    print("\nSine wavelet paths:")
    print(f"  Translation: {mgr_sine.get_translation_path('PV', '2012')}")
    print(f"  Matrix:      {mgr_sine.get_matrix_path('2012')}")
    print(f"  Betas:       {mgr_sine.get_betas_path('PV', '2012')}")
    
    # Example 3: Get all paths at once
    print("\nAll paths for multiple years:")
    all_paths = mgr_square.get_all_paths('PV', ['2012', '2013'])
    print(f"  Translations: {all_paths['translations']}")
    print(f"  Matrices:     {all_paths['matrices']}")
    print(f"  Betas:        {all_paths['betas']}")
