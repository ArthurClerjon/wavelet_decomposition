import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from pathlib import Path
import pickle as pkl
from scipy import sparse
from scipy.sparse.linalg import lsqr
import xlsxwriter


from calc_translations import translate
from utilities import create_directory
from file_manager import WaveletFileManager
from calc_translations import calc_all_translations
import config


def create_directory(path):
    """Create directory if it doesn't exist.
    
    Args:
        path: Can be a file path or directory path.
              If file path (contains '.'), creates parent directory.
              If directory path, creates the directory.
    """
    # Determine if path is a file or directory
    if '.' in os.path.basename(path):
        # It's a file path, get the parent directory
        directory = os.path.dirname(path)
    else:
        # It's a directory path
        directory = path
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"    Created: {directory}")


def wavelet_decomposition_single_TS(TS, year, country_name=None, signal_type=None,
                                    multi_year=None, 
                                    wl_shape='square',
                                    recompute_translation=False,
                                    imp_matrix=True,
                                    dpd=24, dpy=365, 
                                    ndpd=64, vy=6, vw=3, vd=6,
                                    external_translations=None,
                                    reference_signal_type=None):
    """
    Perform wavelet decomposition on a single time series.
    Uses config.py directories for file storage.
    
    Supports using pre-computed translations from another signal (e.g., Load)
    to ensure consistent temporal alignment across multiple signals.
    
    Parameters
    ----------
    TS : array-like
        Input time series data (should be ndpd * dpy points for single year)
    year : str or int
        Year identifier for this data
    country_name : str, optional
        Country/region name (default: 'France')
    signal_type : str, optional
        Signal type: 'PV', 'Wind', 'Consumption' (default: 'Unknown')
    multi_year : list, optional
        List of years if TS contains multiple stacked years
    wl_shape : str, optional
        Wavelet shape: 'square' or 'sine' (default: 'square')
    recompute_translation : bool, optional
        Force recomputation of translations (default: False)
    imp_matrix : bool, optional
        Import existing matrix if available (default: True)
    dpd : int, optional
        Original data points per day (default: 24)
    dpy : int, optional
        Days per year (default: 365)
    ndpd : int, optional
        Interpolated data points per day (default: 64)
    vy : int, optional
        Yearly wavelet levels (default: 6)
    vw : int, optional
        Weekly wavelet levels (default: 3)
    vd : int, optional
        Daily wavelet levels (default: 6)
    external_translations : list or None, optional
        Translations to use for this decomposition.
        None = compute OR import translations for THIS signal
               (normal behavior - uses cached file if exists)
        list = use these provided translations instead
               (skips computation/import entirely, no files created)
        Example: trans_Load (reuse Load's translations for PV)
        Default: None
    reference_signal_type : str or None, optional
        Label for logging (which signal provided translations).
        Only used for display, no effect on calculation.
        Example: 'Consumption'
        Default: None
        
    Returns
    -------
    trans_files : list or None
        List of paths to translation files, or None if external_translations used
    matrix_files : list
        List of matrix file paths
    per_year_betas : dict
        Dictionary of beta coefficients by year
    trans : list
        The translations used (either computed or external) - useful for reuse
        
    Examples
    --------
    # Step 1: Decompose Load (computes translations)
    trans_files_Load, matrix_files_Load, betas_Load, trans_Load = wavelet_decomposition_single_TS(
        Load_single_year, year='2012', signal_type='Consumption')
    
    # Step 2: Use Load's translations for PV (no translation files created)
    trans_files_PV, matrix_files_PV, betas_PV, _ = wavelet_decomposition_single_TS(
        PV_single_year, year='2012', signal_type='PV',
        external_translations=trans_Load, reference_signal_type='Consumption')
    # Note: trans_files_PV will be None
    """
    
    # =========================================================================
    # 1. SETUP DEFAULTS
    # =========================================================================
    
    if country_name is None:
        country_name = 'France'
    if signal_type is None:
        signal_type = 'Unknown'
    
    # Years to process
    if multi_year:
        years = [str(y) for y in multi_year]
    else:
        years = [str(year)]
    
    # Determine if using external translations
    use_external_trans = external_translations is not None
    
    print(f"\n{'='*70}")
    print(f"WAVELET DECOMPOSITION - {signal_type} - {country_name}")
    print(f"{'='*70}")
    print(f"Years to process:   {years}")
    print(f"Data length:        {len(TS)} points")
    print(f"Expected per year:  {ndpd * dpy} points")
    print(f"Wavelet Shape:      {wl_shape}")
    print(f"Import Matrix:      {imp_matrix}")
    print(f"Parameters:         vy={vy}, vw={vw}, vd={vd}")
    
    # Display translation mode
    if use_external_trans:
        ref_name = reference_signal_type if reference_signal_type else "external signal"
        print(f"Translation Mode:   EXTERNAL (from {ref_name})")
        print(f"                    No calculation, no file I/O")
    else:
        print(f"Translation Mode:   COMPUTE/IMPORT (recompute={recompute_translation})")
    
    print(f"{'='*70}\n")
    
    # =========================================================================
    # 2. INITIALIZE FILE MANAGER (controls ALL paths)
    # =========================================================================
    
    file_mgr = WaveletFileManager(
        base_dir=config.RESULTS_DIR,
        region=country_name,
        wl_shape=wl_shape,
        use_nested=True
    )
    
    # =========================================================================
    # 3. GET ALL PATHS FROM FILE MANAGER
    # =========================================================================
    
    # Translation paths - only needed if NOT using external translations
    if not use_external_trans:
        trans_files = [file_mgr.get_translation_path(signal_type, year_str) 
                       for year_str in years]
        path_trans = os.path.dirname(trans_files[0])
    else:
        trans_files = None
        path_trans = None
    
    # Matrix paths (one per year) - always needed
    matrix_files = [file_mgr.get_matrix_path(year_str) 
                    for year_str in years]
    
    # Beta paths (one per year) - always needed
    beta_files = [file_mgr.get_betas_path(signal_type, year_str) 
                  for year_str in years]
    
    # Get directories for display and creation
    matrix_dir = os.path.dirname(matrix_files[0])
    beta_dir = os.path.dirname(beta_files[0])
    
    # =========================================================================
    # 4. CREATE DIRECTORIES
    # =========================================================================
    
    print(f"Creating directories...")
    if path_trans:
        create_directory(path_trans)
    create_directory(matrix_dir)
    create_directory(beta_dir)
    print(f"✅ Directories ready")
    
    print(f"\nPaths from FileManager:")
    if path_trans:
        print(f"  Translation dir: {path_trans}")
        print(f"  Trans files:     {[os.path.basename(f) for f in trans_files]}")
    else:
        print(f"  Translation dir: (none - using external translations)")
    print(f"  Matrix dir:      {matrix_dir}")
    print(f"  Beta dir:        {beta_dir}")
    print(f"  Matrix files:    {[os.path.basename(f) for f in matrix_files]}\n")
    
    # =========================================================================
    # 5. GET TRANSLATIONS (EXTERNAL OR COMPUTE)
    # =========================================================================
    
    if use_external_trans:
        # =====================================================================
        # USE EXTERNAL TRANSLATIONS
        # - No calculation
        # - No file import
        # - No file save
        # =====================================================================
        print("Using external translations...")
        trans = external_translations
        
        # Validate that external translations match number of years
        if len(trans) != len(years):
            raise ValueError(
                f"External translations length ({len(trans)}) does not match "
                f"number of years ({len(years)}). "
                f"Expected one translation per year: {years}"
            )
        
        ref_name = reference_signal_type if reference_signal_type else "external"
        print(f"✅ Using {len(trans)} translations from {ref_name} signal")
        print(f"   Translations: {trans}")
        print(f"   (No translation calculation or file operations)\n")
        
    else:
        # =====================================================================
        # COMPUTE OR IMPORT TRANSLATIONS (original behavior)
        # - Calculates if file doesn't exist or recompute_translation=True
        # - Imports from file if exists and recompute_translation=False
        # - Saves to file after calculation
        # =====================================================================
        print("Processing translations...")
        
        trans = calc_all_translations(
            trans_files=trans_files,
            input_data=TS,
            ndpd=ndpd,
            dpy=dpy,
            wl_shape=wl_shape,
            recompute_translation=recompute_translation
        )
        
        print(f"✅ Translations ready for {len(trans)} years")
        print(f"   Translations: {trans}\n")

    # =========================================================================
    # 6. COMPUTE WAVELET DECOMPOSITION
    # =========================================================================
    
    print(f"Computing wavelet decomposition...")
    
    # Add trailing separator for path concatenation in compute_wavelet_coefficient_betas
    matrix_dir_with_sep = matrix_dir + os.sep
    beta_dir_with_sep = beta_dir + os.sep
    
    stacked_betas, per_year_betas = compute_wavelet_coefficient_betas(
        signal_in=TS,
        vecNb_yr=vy,
        vecNb_week=vw,
        vecNb_day=vd,
        dpy=dpy,
        dpd=ndpd,
        trans=trans,
        path_matrix=matrix_dir_with_sep,
        path_decomposition_results=beta_dir_with_sep,
        wl_shape=wl_shape,
        imp_matrix=imp_matrix,
        years=years
    )
    
    print(f"\n✅ Decomposition complete")
    print(f"   Years processed: {list(per_year_betas.keys())}")
    
    # Validate structure
    for year_str in years:
        if year_str in per_year_betas:
            n_scales = len(per_year_betas[year_str])
            expected = vy + vw + vd + 1  # +1 for offset
            status = "✅" if n_scales == expected else "⚠️"
            print(f"   {status} Year {year_str}: {n_scales} time scales (expected {expected})")
    
    # =========================================================================
    # 7. SAVE INDIVIDUAL BETA FILES
    # =========================================================================
    
    print(f"\nSaving beta coefficients...")
    for i, year_str in enumerate(years):
        beta_file = beta_files[i]
        with open(beta_file, 'wb') as f:
            pkl.dump(per_year_betas[year_str], f)
        print(f"   ✅ Saved: {os.path.basename(beta_file)}")
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"✅ DECOMPOSITION COMPLETE")
    print(f"{'='*70}")
    
    # Show translation source
    if use_external_trans:
        ref_name = reference_signal_type if reference_signal_type else "external"
        print(f"Translations:       From {ref_name} signal")
        print(f"                    (no files created - used external values)")
    else:
        print(f"Translation files:")
        for tf in trans_files:
            status = '✅' if os.path.exists(tf) else '❌'
            print(f"  {status} {tf}")
    
    print(f"\nMatrix files:")
    for mf in matrix_files:
        status = '✅' if os.path.exists(mf) else '❌'
        print(f"  {status} {mf}")
    
    print(f"\nBeta files:")
    for bf in beta_files:
        status = '✅' if os.path.exists(bf) else '❌'
        print(f"  {status} {bf}")
    
    print(f"{'='*70}\n")
    
    # Return trans_files (None if external), matrix_files, betas, and trans
    if use_external_trans:
        return None, matrix_files, per_year_betas, trans
    else:
        return trans_files, matrix_files, per_year_betas, trans


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
"""
# Import required modules
from wavelet_decomposition import wavelet_decomposition_single_TS
import config

# Extract single year of data
year_to_process = '2012'
ndpd = 64
dpy = 365
points_per_year = ndpd * dpy

# Get index for the year
year_index = years.index(year_to_process)
start_idx = year_index * points_per_year
end_idx = (year_index + 1) * points_per_year
TS_single_year = PV_ts[start_idx:end_idx]

# Run decomposition
trans_file, matrix_files, results_betas = wavelet_decomposition_single_TS(
    TS_single_year,
    year=year_to_process,
    multi_year=None,
    country_name='France',
    signal_type='PV',
    wl_shape='square',
    recompute_translation=False,
    imp_matrix=True,
    dpd=24,
    ndpd=64,
    vy=6,
    vw=3,
    vd=6
)

# Check results
print(f"Results betas keys: {results_betas.keys()}")
print(f"Number of time scales: {len(results_betas[year_to_process])}")
"""

def generate_square_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                              trans_vec,
                              path_matrix, matrix_name,
                              import_matrix = True):
    '''
This function generation a matric with square shape wavelets
'''

    #############
    # Translations
    #############
    [transday, transweek, transyear] = trans_vec

    #############
    #base vectors
    #############
    signal_length = dpd*dpy
    if os.path.exists(path_matrix + matrix_name) and import_matrix:
        A_sparse  = sparse.load_npz(path_matrix + matrix_name)
#         A = sparse.csr_matrix.todense(A_sparse)
#         A = np.asarray(A)
        A = [] # only needA_ sparse for the lsqr algorith
        print('Importing matrix A square')
    else:
        print('Computing Matrix A square')

        ###############
        ## Create wavelets Phi and matrix A
        ################
        Phi0 = np.ones((1, signal_length)) / math.sqrt(dpy * dpd)
        ##
        Dt = dpy* dpd # points
        Phi1 = np.zeros(((2**vecNb_yr-1) , signal_length ))
        c = 0
        for k in range(vecNb_yr): # loop over the subdivisions
            i=0
            while i < 2**k: # loop over the time scales
                Phi1[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/math.sqrt(Dt)
                Phi1[c, (2*i+1)* Dt//2 : (i+1)* Dt ] = -1.#/math.sqrt(Dt)
                Phi1[c,:] = translate(Phi1[c,:], -transyear)
                c = c +1
                i= i + 1

            Dt =Dt // 2

        ## Phi2 seconde set of wavelets
        #
        Dt = 7*dpd # points
        Phi2 = np.zeros((52*(2**vecNb_week-1), signal_length))
        c = 0
        for k in range(vecNb_week): # loop over the subdivisions
            i=0
            while i < 2**k*52: # loop over the time scales
                Phi2[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/ math.sqrt(Dt)
                Phi2[c, (2 * i + 1) * Dt // 2: (i +1) * Dt] = -1.# / math.sqrt(Dt)
                Phi2[c,:] = translate(Phi2[c,:], -transweek)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        ## Phi3 seconde set of wavelets
        #
        Dt = dpd # points /day
        Phi3 = np.zeros((dpy*(2**vecNb_day-1) ,signal_length ))
        c = 0
        for k in range(vecNb_day): # loop over the subdivisions
            i=0
            while i < 2**k*dpy: # loop over the time scales
                Phi3[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/ math.sqrt(Dt)
                Phi3[c, (2 * i + 1) * Dt // 2: (i + 1) * Dt] = -1.# / math.sqrt(Dt)
                Phi3[c,:] = translate(Phi3[c, :], -transday)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        A = np.transpose(np.concatenate((Phi0, Phi1, Phi2,Phi3)))

        A_sparse = sparse.csr_matrix(A)
#         Making sure that A is normnalized
        epsilon = 10e-5
#         assert(max(np.sum(np.square(A), axis = 0))-1. < epsilon and 1. - min(np.sum(np.square(A), axis = 0)) < epsilon), 'wavelets are not properly normalized'
        sparse.save_npz(path_matrix + matrix_name, sparse.csr_matrix(A))
    return A_sparse


def sine_function(Dt):
    x = np.linspace(0, 2*np.pi, Dt,endpoint=False)
    sine = np.sin(x)
    return sine

def generate_sine_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                              trans_vec,
                              path_matrix, matrix_name,
                              import_matrix = True):
    '''
    This function generation a matric with sine shape wavelets
    '''
    #############
    # Translations
    #############
    [transday, transweek, transyear] = trans_vec

    #############
    #base vectors
    #############
    signal_length = dpd*dpy
    if os.path.exists(path_matrix + matrix_name) and import_matrix:
        A_sparse  = sparse.load_npz(path_matrix + matrix_name)
#         A = sparse.csr_matrix.todense(A_sparse)
#         A = np.asarray(A)
        A = [] # only needA_ sparse for the lsqr algorith
        print('Importing matrix A sine')
    else:
        print('Computing Matrix A sine')

        ###############
        ## Create wavelets Phi and matrix A
        ################
        Phi0 = np.ones((1, signal_length))# / math.sqrt(dpy * dpd)
        ##
        Dt = dpy* dpd # points
        Phi1 = np.zeros(((2**vecNb_yr-1) , signal_length ))
        c = 0
        for k in range(vecNb_yr): # loop over the subdivisions
            i=0
            while i < 2**k: # loop over the time scales
                Phi1[c, 2*i*Dt//2 : (2*i+2)* Dt//2 ] =  sine_function(Dt)
                Phi1[c,:] = translate(Phi1[c,:], -transyear)
                c = c +1
                i= i + 1

            Dt =Dt // 2

        ## Phi2 seconde set of wavelets
        #
        Dt = 7*dpd # points
        Phi2 = np.zeros((52*(2**vecNb_week-1), signal_length))
        c = 0
        for k in range(vecNb_week): # loop over the subdivisions
            i=0
            while i < 2**k*52: # loop over the time scales
                Phi2[c, i*Dt : (i+1)* Dt ] =  sine_function(Dt)
                Phi2[c,:] = translate(Phi2[c,:], -transweek)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        ## Phi3 seconde set of wavelets
        #
        Dt = dpd # points /day
        Phi3 = np.zeros((dpy*(2**vecNb_day-1) ,signal_length ))
        c = 0
        for k in range(vecNb_day): # loop over the subdivisions
            i=0
            if Dt <= 4:
                while i < 2 ** k*dpy:  # With 4 or two points cannot create a sinus
                    Phi3[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.  # /math.sqrt(Dt)
                    Phi3[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.  # /math.sqrt(Dt)
                    Phi3[c, :] = translate(Phi3[c, :], -transday)
                    c = c + 1
                    i = i + 1

                Dt = Dt // 2
            else:
                while i < 2**k*dpy: # loop over the time scales
                    Phi3[c, i*Dt : (i+1)* Dt ] =  sine_function(Dt)
                    Phi3[c, :] = translate(Phi3[c, :], -transday)
                    c = c +1
                    i= i + 1

                Dt = Dt // 2

        A = np.transpose(np.concatenate((Phi0, Phi1, Phi2,Phi3)))

        A_sparse = sparse.csr_matrix(A)
#         Making sure that A is normnalized
        epsilon = 10e-5
#         assert(max(np.sum(np.square(A), axis = 0))-1. < epsilon and 1. - min(np.sum(np.square(A), axis = 0)) < epsilon), 'wavelets are not properly normalized'
        sparse.save_npz(path_matrix + matrix_name, sparse.csr_matrix(A))
    return A_sparse


def beta_decomposition(A_sparse, signal_in):
    # A_sparse = sparse.csr_matrix(A)
    beta_lsqr = lsqr(A_sparse, signal_in, damp=0.001, atol=0, btol=0, conlim=0)[0]
    # Damping coefficient has to be smaller than 0.1. when damp gets big, we loose the reconstruction ( from damp=0.1). When too small, we loose linearity
    return beta_lsqr



def compute_betas(time_series,  stacked_data,
                 vecNb_yr, vecNb_week, vecNb_day, dpy, dpd, years,
                 trans,
                 path_matrix,
                 beta_path, wl_shape, imp_matrix = True ):
    '''
    This function:
    - Compute betas for each imput signal
    - Reshape betas from a 1D-array to a dictionnary with N (15) time scales rows
    - Translate in the othert directions the beta
    - Export in an excel document with different sheets
    - Concatenate all years in a signle sheet
    - Export concatenated betas in a disctionnary, with input signals as jeys of the dictionnary
    - wl_shape : takes 2 values, either square ore sine
    '''
    #
    signal_length = dpy * dpd

    stacked_betas = {}
    workbook2 = xlsxwriter.Workbook(beta_path + 'betas_stacked.xlsx') #All years are stacked in this excel file. One sheet per input siganm

    saved_sheets = {}
    for signal_type in time_series:

        signal_in = stacked_data[signal_type]
#
# 1) ----- Compute betas for a given input signal -------
# -------- returns a 1D array with N years stacked
        betas = []
        for i, year in enumerate(years):
            matrix_name = "A_"+ year + ".npz"
            print(path_matrix + matrix_name)
            if wl_shape == 'square':
                A_sparse = generate_square_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                    trans[i],
                                                    path_matrix, matrix_name,
                                                    import_matrix = imp_matrix)
                print('Square sparsee matrix or year '+ year +' has been imported')
            elif wl_shape == 'sine':
                A_sparse = generate_sine_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                    trans[i],
                                                    path_matrix, matrix_name,
                                                    import_matrix = imp_matrix)
                print('Sine sparse matrix or year '+ year +' has been imported')
            else:
                print('The type of wavelet is not defined. Please type "square" or "sine"')

            betas.append(beta_decomposition(A_sparse, signal_in[signal_length*i:signal_length*(i+1)]) )

        #
        # -------- Open Excel file ----------
        workbook = xlsxwriter.Workbook(beta_path + 'betas_'+ signal_type + '.xlsx')
        row = 0
        saved_sheets[signal_type] = {}
#
# 2) ----- Reshape betas in a list of 16 time scales -------
# -------- Time scales icludes the offset value
        for i,beta in enumerate(betas):
            saved_sheets[signal_type][years[i]] = []

            worksheet = workbook.add_worksheet(str(years[i]))

            # -- Initialization --
            len_max = dpy *(2**vecNb_day-1)
            newsize = dpy *(2**vecNb_day-1)
            total_vec = vecNb_day+vecNb_week+vecNb_yr # number of time scales
            sheet = []

            beta_offset =[beta[0]]
            beta_year = beta[1 : 1+ 2**vecNb_yr-1] # all betas comming from the yearly motheer wavelet
            beta_week = beta[1+ 2**vecNb_yr - 1: 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1)]
            beta_day = beta[ 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1):  1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1) + dpy * (2**vecNb_day-1)]
            #

            sheet.append(beta_offset)
            # --- Year vec ------
            for k in range(vecNb_yr):
                sheet.append(beta_year[2 ** k - 1: 2 ** (k + 1) - 1].tolist())
            # --- Week vec ------
            for k in range(vecNb_week):
                sheet.append(beta_week[52 * (2 ** k - 1): 52 * (2 ** (k + 1) - 1)].tolist())
            # --- Day vec ------
            for k in range(vecNb_day):
                sheet.append(beta_day[dpy * (2 ** k - 1):dpy * (2 ** (k + 1) - 1)].tolist())


            #reverse the list order
            sheet = [sh[::-1] for sh in reversed(sheet) ]

            saved_sheets[signal_type][years[i]] = sheet

            for col, data in enumerate(sheet):
                worksheet.write_column(row, col, data)

            if len(saved_sheets[signal_type][years[i]][-1])>1:
                print('error1')

        workbook.close()
#
# 3) ----- Stack all betas in a 16 dimensions time scale list  -------

        worksheet2 = workbook2.add_worksheet(signal_type)
        row = 0
        # Initialization
        stacked_sheet = [None] * len(saved_sheets[signal_type][years[0]])

        for ts in range(len(stacked_sheet)):
            tmp = []
            for i in range(len(years)):
                tmp.extend(saved_sheets[signal_type][years[i]][ts])

            stacked_sheet[ts] = tmp

        for col, data in enumerate(stacked_sheet):
            worksheet2.write_column(row, col, data)

        stacked_betas[signal_type] = stacked_sheet
        #
    workbook2.close()

    return stacked_betas, saved_sheets

def preplotprocessing(vecNb_yr, vecNb_week , vecNb_day, ndpd, dpy,
                    signal_type, year, years, time_scales, saved_sheets, matrix):
    '''
    Preprocess waveley sheets for plot_betas_heatmap() function
This function takes as imputs:
    - saved_sheets
    - signal_type : 'Consommation' or 'Eolien'...
    - list of years included. e.g. ['2012', '2013',...]
    - year : e.g. '2012'

What it does:
    - Reshape betas as a DataFrame with row of equal size. Ready to be ploted
    '''
    #
    # translation
    assert(years[years.index(year)] == year), 'Index error between the translation year and the data'
    #
    # Initialization
    Nb_vec = vecNb_yr + vecNb_week + vecNb_day
    max_nb_betas = dpy*ndpd

    assert(Nb_vec+1 == len(saved_sheets[signal_type][year]) ), 'There is not the right number of time scales' # +1 stands for the offset value
    # Create an empty DataFrame (nan)
    df = pd.DataFrame(np.nan, index=range(Nb_vec), columns=range(max_nb_betas)).transpose()

    for k,ts in enumerate(time_scales):
        new_vec = reconstruct(time_scales, [ts],
                    matrix, saved_sheets[signal_type][year], "Consommation électrique Française normalisée, 2013",
                    xmin=0, xmax=365,
                    dpy=dpy, dpd=ndpd,
                    add_offset=False)
        plt.close()
        df[k] = pd.DataFrame({'betas':new_vec})
    return df


def reconstruct(time_scales, reconstructed_time_scales,
                matrix, beta_sheet, title,
                xmin=0, xmax=365,
                dpy=365, dpd=64,
                add_offset=True, plot = True):
    '''
    This function reconstruct time series for given time scales. It only workd for 1 year signals.
    Takes as inputs :
     - time_scales :  [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380., 8760.] # cycles length, in hours
     - reconstructed_time_scales : e.g. [24] if you want to reconstructa signal filtered with the daily wavelets
     - matrix, square or sine, with the year
     - beta_sheets : A one year list of betas ordered by time scales
     - title : of the figure
     - dpy : days per year. 365 default value
     - dpd : data per day: 64 defauld value
     - add_offset : If you want to add or remove the offset of the siganl
    '''

    #     time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380., 8760.] # cycles length, in hours
    # reconstructed_time_scales = time_scales
    # Concat time scales
    concat_betas = []

    # Convert to sets for faster lookup and better float comparison
    reconstructed_set = set(reconstructed_time_scales)

    for i, ts in enumerate(time_scales):
        # Use approximate float comparison (within tolerance)
        is_selected = any(abs(ts - rts) < 1e-6 for rts in reconstructed_set)

        if is_selected:
            concat_betas.extend(beta_sheet[i])
        else:
            # Add zeros for non-selected time scales
            concat_betas.extend([0.] * len(beta_sheet[i]))

    if add_offset:
        concat_betas.extend(beta_sheet[-1])
    else:
        concat_betas.extend([0.])

    # Perform matrix multiplication (works with both dense and sparse matrices)
    reconstructed = matrix.dot(concat_betas[::-1])

    # PLots options
    if plot :
        sns.set()
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.})
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.set_palette("colorblind")  # set colors palettte
        #
        time = np.linspace(0, dpy, dpy * dpd)
        fig = plt.figure()
        fig.set_size_inches(10, 8)
        plt.plot(time, reconstructed)
        plt.xlim(xmin, xmax)
        plt.xlabel('Days')
        plt.ylabel('Power')
        plt.title(title)
        plt.show()

    return reconstructed


def stack_betas(saved_sheets, time_series, chosen_years):
    '''
    This function returns stacked betas for chosen number of years
    It takes as arguments :
    - saved_sheets: a dictionnary with signal types as keyx (e.g. "Consommation") and year then
    For instance saved_sheets['Consommation']["2012"] returns a list of N time scales
    - chosen years: the years picked up amoung the "years" imported
    '''
    stacked_betas = {}
    for signal_type in time_series:
        stacked_sheet = [None] * len(saved_sheets[signal_type][chosen_years[0]])

        for ts in range(len(stacked_sheet)):
            tmp = []
            for yr in chosen_years:
                tmp.extend(saved_sheets[signal_type][yr][ts])

            stacked_sheet[ts] = tmp
        print(signal_type)

        stacked_betas[signal_type] = stacked_sheet
    return stacked_betas



def reconstruct_per_ts(A, trans, signal, do_trans, add_offset=False):
    '''
    This function proceed to a wavelet decomposition of an input signal for each time scales
    It returns alist of 15 decomposed signals
    :param A:
    :param trans:
    :param signal:
    :param do_trans:
    :param add_offset:
    :return:
    '''

    # A = generate_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, time, dpd, trans)
    A_sparse = sparse.csr_matrix(A)
    if type(signal) is dict:
        beta = {}
        for key in signal.keys():
            beta[key] = beta_decomposition(A_sparse, signal[key], trans)
            check_orthogonamoty(beta[key])
    if type(signal) is np.ndarray:
        beta = beta_decomposition(A_sparse, signal, trans)
        # check_orthogonamoty(beta) #todo: the function is not made properly

    reconstructed_signal = {}
    c = 0
    # this manipulation with c and i has to be done because in this reconstruct_signal function, times scales go from year to hour, whereas we need the for hour to year.
    for i in reversed(range(len(time_scale))):
        use_beta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        use_beta[i] = 1

        if type(signal) is dict:
            tmp = {}
            for key in beta.keys():
                # tmp[str(key)] = interpolate(reconstruct_signal(A, beta[key], use_beta, do_trans, trans), time_scale[::-1][i], dpd, Nyears_signal)[0]
                tmp[str(key)] = \
                    reconstruct_signal(A, beta[key], use_beta, do_trans, trans,  add_offset)
            reconstructed_signal[c] = tmp

        if type(signal) is np.ndarray:
            reconstructed_signal[c] = reconstruct_signal(A, beta, use_beta, do_trans, trans, add_offset)
            #reconstructed_signal[c] =  interpolate(reconstruct_signal(A, beta, use_beta, do_trans, trans), time_scale[::-1][i], dpd, Nyears_signal)[0]
        c = c + 1
    return reconstructed_signal

def compute_wavelet_coefficient_betas(signal_in,
                 vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                 trans,
                 path_matrix,
                 path_decomposition_results, wl_shape, imp_matrix = True,
                  years = None ):
    '''
    This function:
    - Compute the wavelet coefficients (named betas).
    - Reshape betas from a 1D-array to a dictionnary with N (15) time scales rows
    - Translate in the othert directions the beta
    - Export in an excel document with different sheets
    - Concatenate all years in a signle sheet
    - Export concatenated betas in a disctionnary, with input signals as jeys of the dictionnary
    - wl_shape : takes 2 values, either square ore sine
    - years is a list of year of the input time serie. For istance ['2017' , '2018']
      if years = None, it would be replaced by years = ['0', '1']

    It returns :
    - 2 Excel files in the directory ath_decomposition_results :
        * The first gives all decompositions coefficients (names betas) stacked per time sclale. Rhus, if the input data last 2 years, there is two coefficient for the time scale 'year'.
        * The second excel file returns the results of each year decomposition in a different sheet.

    - Results array...
    '''

    create_directory(path_decomposition_results)
    #
    signal_length = len(signal_in)
    assert (signal_length % (dpy * dpd) == 0), 'The signal length is not an integer number of years.'

    N_years = int(signal_length/(dpy * dpd))
    if years is None:
        years = [str(y) for y in range(N_years)] 

    workbook2 = xlsxwriter.Workbook(path_decomposition_results + 'results_betas_stacked.xlsx') #The results of the decomposition of each year are stacked in this sheet. 

# 1) ----- Compute betas for a given input signal -------
# -------- returns a 1D array with N years stacked
    betas = []
    for i, year in enumerate(years):
        #print(i,year)
        matrix_name = "A_"+ year + ".npz"
        # print(path_matrix + matrix_name)
        if wl_shape == 'square':
            # print(trans)
            A_sparse = generate_square_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                trans[i],
                                                path_matrix, matrix_name,
                                                import_matrix = imp_matrix)
            print('Square sparsee matrix on year '+ year +' has been imported')
        elif wl_shape == 'sine':
            A_sparse = generate_sine_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                trans[i],
                                                path_matrix, matrix_name,
                                                import_matrix = imp_matrix)
            print('Sine sparse matrix on year '+ year +' has been imported')
        else:
            print('The type of wavelet is not defined. Please type "square" or "sine"')

        start_index = int(signal_length/N_years*i)
        end_index = int(signal_length/N_years*(i+1))
        betas.append(perform_wavelet_decomposition(A_sparse, signal_in[start_index:end_index]) )

    #
    # -------- Open Excel file ----------
    workbook = xlsxwriter.Workbook(path_decomposition_results + 'results_per_year_betas.xlsx')
    row = 0
    per_year_betas = {}
#
# 2) ----- Reshape betas in a list of 16 time scales -------
# -------- Time scales icludes the offset value
    for i,beta in enumerate(betas):
        per_year_betas[years[i]] = []

        worksheet = workbook.add_worksheet(str(years[i]))

        # -- Initialization --
        len_max = dpy *(2**vecNb_day-1)
        newsize = dpy *(2**vecNb_day-1)
        total_vec = vecNb_day+vecNb_week+vecNb_yr # number of time scales
        sheet = []

        beta_offset =[beta[0]]
        beta_year = beta[1 : 1+ 2**vecNb_yr-1] # all betas comming from the yearly motheer wavelet
        beta_week = beta[1+ 2**vecNb_yr - 1: 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1)]
        beta_day = beta[ 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1):  1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1) + dpy * (2**vecNb_day-1)]
        #

        sheet.append(beta_offset)
        # --- Year vec ------
        for k in range(vecNb_yr):
            sheet.append(beta_year[2 ** k - 1: 2 ** (k + 1) - 1].tolist())
        # --- Week vec ------
        for k in range(vecNb_week):
            sheet.append(beta_week[52 * (2 ** k - 1): 52 * (2 ** (k + 1) - 1)].tolist())
        # --- Day vec ------
        for k in range(vecNb_day):
            sheet.append(beta_day[dpy * (2 ** k - 1):dpy * (2 ** (k + 1) - 1)].tolist())


        # Reverse the list order
        sheet = [sh[::-1] for sh in reversed(sheet) ]

        per_year_betas[years[i]] = sheet

        for col, data in enumerate(sheet):
            worksheet.write_column(row, col, data)

        if len(per_year_betas[years[i]][-1])>1:
            print('error1')

    workbook.close()
#
# 3) ----- Stack all betas in a 16 dimensions time scale list  -------

    worksheet2 = workbook2.add_worksheet('Wavelet decomposition results')
    row = 0
    # Initialization
    stacked_sheet = [None] * len(per_year_betas[years[0]])

    for ts in range(len(stacked_sheet)):
        tmp = []
        for i in range(len(years)):
            tmp.extend(per_year_betas[years[i]][ts])

        stacked_sheet[ts] = tmp

    for col, data in enumerate(stacked_sheet):
        worksheet2.write_column(row, col, data)

    stacked_betas = stacked_sheet
    #
    workbook2.close()

    return stacked_betas, per_year_betas

def perform_wavelet_decomposition(A_sparse, signal_in):
    # A_sparse = sparse.csr_matrix(A)
    beta_lsqr = lsqr(A_sparse, signal_in, damp=0.001, atol=0, btol=0, conlim=0)[0]
    # Damping coefficient has to be smaller than 0.1. when damp gets big, we loose the reconstruction ( from damp=0.1). When too small, we loose linearity
    return beta_lsqr


