import numpy as np
import pickle as pkl
import os
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

def sine_function(Dt):
    x = np.linspace(0, 2*np.pi, Dt, endpoint = False)
    sine = np.sin(x)
    return sine

def calc_trans(ndpd, dpy, input_data, wl_shape):
    '''
    Compute best translations for each years of the input data
    :param ndpd: data per day
    :param dpy: days per year
    :param input_data: stacked time series (1D vector)
    :param wl_shape: 'square' or 'sine_function' : shape of the wavelet
    :return: a list of translations for each uear
    '''
    veclength = ndpd*dpy
    Nyears = int(len(input_data)/veclength)
    assert(dpy*ndpd % len(input_data) ),'Number of years and points are not consistent'
    assert(wl_shape != 'sine' or wl_shape != 'square'), 'Shape error. must be either square or sine_function'
    trans = []
    for k in range(Nyears):
        signal_in = input_data[k*veclength: (k+1)*veclength]
        # Year
        # Creat year mother waveley
        #
        Dt = dpy * ndpd
        signal_length = dpy * ndpd
        #
        vec_year = np.zeros((1, signal_length))
        if wl_shape == 'square':
            vec_year[0, 0:  Dt // 2] = 1.  # /math.sqrt(Dt)
            vec_year[0, Dt // 2:  Dt] = -1.  # /math.sqrt(Dt)
        if wl_shape == 'sine':
            vec_year[0, :] =  sine_function(Dt)
        vec_year_sparse = sparse.csr_matrix(np.transpose(vec_year))

        best_trans_year = calc_best_trans(vec_year, vec_year_sparse, signal_in, ndpd, dpy)

        # ----------------------------
        # Week
        # Creat week mother wavelets
        Dt = 7 * ndpd  # points
        vec_week = np.zeros((52, signal_length))
        c = 0
        i = 0
        while i < 52:  # loop over the time scales
            if wl_shape == 'square':
                vec_week[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                vec_week[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            if wl_shape == 'sine':
                vec_week[c, i*Dt : (i+1)*Dt ] = sine_function(Dt)
            c = c + 1
            i = i + 1

        vec_week_sparse = sparse.csr_matrix(np.transpose(vec_week))

        best_trans_week = calc_best_trans(vec_week, vec_week_sparse, signal_in, ndpd, dpy)

        # ----------------------------
        # Days
        # Creat daymother wavelets
        Dt = ndpd  # points /day
        vec_day = np.zeros((dpy, signal_length))
        c = 0
        i = 0
        while i < dpy:  # loop over the time scales
            if wl_shape == 'square':
                vec_day[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                vec_day[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            if wl_shape == 'sine':
                vec_day[c, i*Dt : (i+1)*Dt] = sine_function(Dt)
            c = c + 1
            i = i + 1
        vec_day_sparse = sparse.csr_matrix(np.transpose(vec_day))

        best_trans_day = calc_best_trans(vec_day, vec_day_sparse, signal_in, ndpd, dpy)

        print([best_trans_day, best_trans_week, best_trans_year])
        trans.append( [best_trans_day, best_trans_week, best_trans_year] )
    return trans

def translate(data, d):
    while d < 0: #what is the use of this ?
        d = d + data.size
    tmp = np.zeros(data.size)
    for i in range(data.size):
        tmp[i] = data[(i + d) % data.size]
    return tmp

def calc_residue(data, wavelets, sparse_wavelets):
    '''
    Calculate the baseline residue, before optimizing the position of the wavelet on the signal.

    '''
    data_no_mean = data - np.mean(data)
    betas = lsqr(sparse_wavelets, data_no_mean, damp=0.001, atol=0, btol=0, conlim=0)[0]
    
    for i in range(wavelets.shape[0]):
        data_no_mean = data_no_mean - betas[i]*wavelets[i,:]
    residue = np.sum(data_no_mean*data_no_mean)
    print('Residue')
    print(residue)
    return residue


def compute_single_year_translation(signal_single_year, ndpd, dpy, wl_shape):
    """
    Compute optimal translations for a SINGLE year of data.
    
    This is a PURE COMPUTATION function - no file I/O.
    File management is handled by the caller.
    
    Parameters
    ----------
    signal_single_year : array-like
        Single year time series data (length = ndpd * dpy)
    ndpd : int
        Number of data points per day
    dpy : int
        Days per year (typically 365)
    wl_shape : str
        Wavelet shape ('square' or 'sine')
        
    Returns
    -------
    list
        [trans_day, trans_week, trans_year] - optimal translation offsets
    """
    signal_length = dpy * ndpd
    
    # Validate input
    if len(signal_single_year) != signal_length:
        raise ValueError(
            f"Signal length {len(signal_single_year)} != expected {signal_length} "
            f"(ndpd={ndpd} * dpy={dpy})"
        )
    
    # Remove mean for correlation computation
    signal_centered = signal_single_year - np.mean(signal_single_year)
    
    # =========================================================================
    # YEAR WAVELET TRANSLATION
    # =========================================================================
    Dt = dpy * ndpd
    vec_year = np.zeros((1, signal_length))
    if wl_shape == 'square':
        vec_year[0, 0:Dt // 2] = 1.
        vec_year[0, Dt // 2:Dt] = -1.
    elif wl_shape == 'sine':
        vec_year[0, :] = sine_function(Dt)
    
    best_trans_year = best_translation_year(vec_year, signal_centered)
    
    # =========================================================================
    # WEEK WAVELET TRANSLATION
    # =========================================================================
    Dt = 7 * ndpd  # points per week
    vec_week = np.zeros((52, signal_length))
    for i in range(52):
        if wl_shape == 'square':
            vec_week[i, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
            vec_week[i, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
        elif wl_shape == 'sine':
            vec_week[i, i * Dt: (i + 1) * Dt] = sine_function(Dt)
    
    best_trans_week = best_translation_week(vec_week, signal_centered, ndpd)
    
    # =========================================================================
    # DAY WAVELET TRANSLATION
    # =========================================================================
    Dt = ndpd  # points per day
    vec_day = np.zeros((dpy, signal_length))
    for i in range(dpy):
        if wl_shape == 'square':
            vec_day[i, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
            vec_day[i, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
        elif wl_shape == 'sine':
            vec_day[i, i * Dt: (i + 1) * Dt] = sine_function(Dt)
    
    best_trans_day = best_translation_day(vec_day, signal_centered, ndpd)
    
    return [best_trans_day, best_trans_week, best_trans_year]


def calc_all_translations(trans_files, input_data, ndpd, dpy, wl_shape,
                          recompute_translation=False):
    """
    Compute or load translations for all years.
    
    IMPORTANT: File paths are provided by the caller (from FileManager).
    This function does NOT construct paths - it only uses what's given.
    
    Parameters
    ----------
    trans_files : list of str
        List of file paths for each year's translation file.
        Paths should be provided by WaveletFileManager.get_translation_path()
        Example: ['results/France/translations/trans_France_PV_2012.pkl',
                  'results/France/translations/trans_France_PV_2013.pkl']
    input_data : array-like
        Stacked time series (1D vector containing N years)
    ndpd : int
        Number of data points per day
    dpy : int
        Days per year
    wl_shape : str
        Wavelet shape: 'square' or 'sine'
    recompute_translation : bool, optional
        If True, recompute even if file exists (default: False)
        
    Returns
    -------
    trans : list
        List of [trans_day, trans_week, trans_year] for each year
        
    Example
    -------
    >>> from file_manager import WaveletFileManager
    >>> 
    >>> # FileManager controls all paths
    >>> file_mgr = WaveletFileManager(region='France')
    >>> years = ['2012', '2013', '2014']
    >>> trans_files = [file_mgr.get_translation_path('PV', y) for y in years]
    >>> 
    >>> # calc_all_translations just uses those paths
    >>> trans = calc_all_translations(
    ...     trans_files=trans_files,
    ...     input_data=my_data,
    ...     ndpd=64, dpy=365,
    ...     wl_shape='square'
    ... )
    """
    veclength = ndpd * dpy
    Nyears = int(len(input_data) / veclength)
    
    # Validate inputs
    if len(input_data) % (dpy * ndpd) != 0:
        raise ValueError('Signal length is not an integer number of years.')
    
    if wl_shape not in ['sine', 'square']:
        raise ValueError(f"wl_shape must be 'square' or 'sine', got '{wl_shape}'")
    
    if len(trans_files) != Nyears:
        raise ValueError(
            f"trans_files has {len(trans_files)} entries but data contains {Nyears} years"
        )
    
    # Process each year
    trans = []
    
    for k in range(Nyears):
        trans_file = trans_files[k]
        
        # Ensure directory exists
        trans_dir = os.path.dirname(trans_file)
        if trans_dir:
            os.makedirs(trans_dir, exist_ok=True)
        
        # Check if file exists for this year
        if os.path.exists(trans_file) and not recompute_translation:
            print(f"  Loading: {os.path.basename(trans_file)}")
            with open(trans_file, 'rb') as f:
                year_trans = pkl.load(f)
        else:
            # Extract single year data
            signal_single_year = input_data[k * veclength: (k + 1) * veclength]
            
            # Compute translation using core function
            reason = "recompute=True" if recompute_translation else "file not found"
            print(f"  Computing ({reason}): {os.path.basename(trans_file)}")
            
            year_trans = compute_single_year_translation(
                signal_single_year, ndpd, dpy, wl_shape
            )
            print(f"    -> day={year_trans[0]}, week={year_trans[1]}, year={year_trans[2]}")
            
            # Save to the path provided by FileManager
            with open(trans_file, 'wb') as f:
                pkl.dump(year_trans, f)
            print(f"    -> Saved")
        
        trans.append(year_trans)
    
    return trans


# #
#  #                         ndpd, dpy, input_data, wl_shape, 
#                           recompute_translation=False,
#                           save_per_year=False,
#                           years=None):
#     """
#     Compute best translations for each year of the input data.
    
#     Parameters
#     ----------
#     path_trans : str
#         Directory path where translation files will be saved
#     translation_name : str
#         Base name for the translation file (e.g., 'France_PV')
#     ndpd : int
#         Number of data points per day
#     dpy : int
#         Days per year
#     input_data : array-like
#         Stacked time series (1D vector containing N years)
#     wl_shape : str
#         Wavelet shape: 'square' or 'sine'
#     recompute_translation : bool, optional
#         If True, recompute even if file exists (default: False)
#     save_per_year : bool, optional
#         If True, save separate file per year (default: False for backward compatibility)
#     years : list, optional
#         List of year identifiers. Required if save_per_year=True.
        
#     Returns
#     -------
#     If save_per_year=False (legacy mode):
#         filename_pkl : str
#             Path to the single translation file
#         trans : list
#             List of [trans_day, trans_week, trans_year] for each year
            
#     If save_per_year=True (new mode):
#         trans_files : list
#             List of paths to per-year translation files
#         trans : list
#             List of [trans_day, trans_week, trans_year] for each year
#     """
#     veclength = ndpd * dpy
#     Nyears = int(len(input_data) / veclength)
    
#     # Validate inputs
#     assert len(input_data) % (dpy * ndpd) == 0, \
#         'Signal length is not an integer number of years.'
#     assert wl_shape in ['sine', 'square'], \
#         f"wl_shape must be 'square' or 'sine', got '{wl_shape}'"
    
#     if save_per_year and years is None:
#         # Auto-generate year names if not provided
#         years = [str(i) for i in range(Nyears)]
    
#     if save_per_year and len(years) != Nyears:
#         raise ValueError(
#             f"years list has {len(years)} entries but data contains {Nyears} years"
#         )
    
#     # Ensure directory exists
#     os.makedirs(path_trans, exist_ok=True)
    
#     # =========================================================================
#     # MODE 1: Per-Year Files (NEW)
#     # =========================================================================
#     if save_per_year:
#         trans = []
#         trans_files = []
        
#         for k in range(Nyears):
#             year_str = years[k]
#             filename_year = os.path.join(path_trans, f'trans_{translation_name}_{year_str}.pkl')
#             trans_files.append(filename_year)
            
#             # Check if file exists for this year
#             if os.path.exists(filename_year) and not recompute_translation:
#                 print(f"  Year {year_str}: Loading from {os.path.basename(filename_year)}")
#                 with open(filename_year, 'rb') as f:
#                     year_trans = pkl.load(f)
#             else:
#                 # Extract single year data
#                 signal_single_year = input_data[k * veclength: (k + 1) * veclength]
                
#                 # Compute translation using core function
#                 print(f"  Year {year_str}: Computing translation...")
#                 year_trans = compute_single_year_translation(
#                     signal_single_year, ndpd, dpy, wl_shape
#                 )
#                 print(f"    -> day={year_trans[0]}, week={year_trans[1]}, year={year_trans[2]}")
                
#                 # Save per-year file
#                 with open(filename_year, 'wb') as f:
#                     pkl.dump(year_trans, f)
#                 print(f"    -> Saved to {os.path.basename(filename_year)}")
            
#             trans.append(year_trans)
        
#         return trans_files, trans
    
#     # =========================================================================
#     # MODE 2: Single File for All Years (LEGACY - backward compatible)
#     # =========================================================================
#     else:
#         filename_pkl = os.path.join(path_trans, f'results_translation_{translation_name}.pkl')
        
#         # Check if file exists and has correct number of years
#         if os.path.exists(filename_pkl) and not recompute_translation:
#             with open(filename_pkl, 'rb') as f:
#                 trans = pkl.load(f)
            
#             if len(trans) == Nyears:
#                 print(f"Loading existing translation file: {filename_pkl}")
#                 return filename_pkl, trans
#             else:
#                 print(f"File exists but has {len(trans)} years, need {Nyears}. Recomputing...")
        
#         # Compute translations for all years
#         print(f"Computing translations for {Nyears} years...")
#         trans = []
        
#         for k in range(Nyears):
#             signal_single_year = input_data[k * veclength: (k + 1) * veclength]
            
#             # Use core computation function
#             year_trans = compute_single_year_translation(
#                 signal_single_year, ndpd, dpy, wl_shape
#             )
#             print(f"  Year {k}: day={year_trans[0]}, week={year_trans[1]}, year={year_trans[2]}")
#             trans.append(year_trans)
        
#         # Save to single file
#         with open(filename_pkl, 'wb') as f:
#             pkl.dump(trans, f)
#         print(f"Saved to: {filename_pkl}")
        
#         return filename_pkl, trans


# #def calc_all_translations(path_trans, translation_name, 
                          
                          
#                                ndpd, dpy, input_data, wl_shape, 
#                                recompute_translation= False):
    
#     '''
#     Compute best translations for each years of the input data
#     :param ndpd: data per day
#     :param dpy: days per year
#     :param input_data: stacked time series (1D vector)
#     :param wl_shape: 'square' or 'sine_function' : shape of the wavelet
#     :return: a list of translations for each uear
#     '''

#     veclength = ndpd*dpy
#     Nyears = int(len(input_data)/veclength)

#     signal_length = len(input_data)
#     assert (signal_length % (dpy * ndpd) == 0), 'The signal length is not an integer number of years.'
#     assert(wl_shape != 'sine' or wl_shape != 'square'), 'Shape error. must be either square or sine_function'

#     # Check if the file exists and is consistent with the number of years of the input signal. If not, recomputing the translations
#     filename_pkl = os.path.join(os.getcwd(), path_trans, 'results_translation_'+ translation_name +'.pkl')

#     if os.path.exists(filename_pkl) and not recompute_translation:
#         # Load the data from the 'results_translation.pkl' file if its size is consistent with the number of year of the input signal
#         with open(filename_pkl, 'rb') as file:
#             trans = pkl.load(file)
#         if len(trans) == Nyears :
#             print(f"Loading existing translation file: {filename_pkl}")
#     else:
#         # File does not exist, so compute the translation
#         print("Computing translation...")

#         trans = []
#         for k in range(Nyears):
#             signal_in = input_data[k*veclength: (k+1)*veclength] #get the data for one year only to perform the translation

#             # Year
#             # Creat year mother wavelet
            
#             Dt = dpy * ndpd
#             signal_length = dpy * ndpd
            
#             vec_year = np.zeros((1, signal_length))
#             if wl_shape == 'square':
#                 vec_year[0, 0:  Dt // 2] = 1.  # /math.sqrt(Dt)
#                 vec_year[0, Dt // 2:  Dt] = -1.  # /math.sqrt(Dt)
#             if wl_shape == 'sine':
#                 vec_year[0, :] =  sine_function(Dt)

#             best_trans_year = best_translation_year(vec_year, signal_in)
#             # ----------------------------
#             # Week
#             # Create week mother wavelets
#             Dt = 7 * ndpd  # points
#             vec_week = np.zeros((52, signal_length))
#             c = 0
#             i = 0

#             for i in range(52): 
#                 c=i
#                 if wl_shape == 'square':
#                     vec_week[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
#                     vec_week[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
#                 if wl_shape == 'sine':
#                     vec_week[c, i*Dt : (i+1)*Dt ] = sine_function(Dt)

#             # vec_week_sparse = sparse.csr_matrix(np.transpose(vec_week))

#             best_trans_week = best_translation_week(vec_week, signal_in, ndpd)

#             # ----------------------------
#             # Days
#             # Creat daymother wavelets
#             Dt = ndpd  # points /day
#             vec_day = np.zeros((dpy, signal_length))
#             c = 0
#             i = 0
#             for i in range(dpy):  # loop over the time scales
#                 if wl_shape == 'square':
#                     vec_day[i, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
#                     vec_day[i, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
#                 if wl_shape == 'sine':
#                     vec_day[i, i*Dt : (i+1)*Dt] = sine_function(Dt)
#             # vec_day_total = np.sum(vec_day, axis =0)
#             best_trans_day = best_translation_day(vec_day, signal_in, ndpd)

#             print(f"Best translation day = {best_trans_day}")
#             print(f"Best translation week = {best_trans_week}")
#             print(f"Best translation year = {best_trans_year}")
#             trans.append([best_trans_day,best_trans_week, best_trans_year] )

#             # Save the results of the translation in the 'translation/' directory
#             with open(filename_pkl, 'wb') as file:
#                 pkl.dump(trans, file)
#     return filename_pkl, trans

def best_translation_week(wavelet, input_data, ndpd): #find translation where scalar product is maximal of one wavelet
    
    input_data = input_data-input_data.mean()
    total_vec_week = np.sum(wavelet, axis =0)
    best_scalar_product = np.dot(total_vec_week, input_data)
    t0=0
    for j in range(7*ndpd):
        vec_week_shifted = np.roll(total_vec_week, j)  # loop over the time scales
        scalar_product = np.dot(vec_week_shifted, input_data)
        print(scalar_product)
        if scalar_product>best_scalar_product:
            best_scalar_product = scalar_product
            t0 = j
    return t0

def best_translation_day(wavelet, input_data, ndpd): #find translation where scalar product is maximal of one wavelet
    
    input_data = input_data-input_data.mean()
    total_vec_day = np.sum(wavelet, axis =0)
    best_scalar_product = np.dot(total_vec_day, input_data)
    t0=0
    for j in range(ndpd):
        vec_day_shifted = np.roll(total_vec_day, j)  # loop over the time scales
        scalar_product = np.dot(vec_day_shifted, input_data)
        print(scalar_product)
        if scalar_product>best_scalar_product:
            best_scalar_product = scalar_product
            t0 = j
    return t0

def best_translation_year(wavelet, input_data): #find translation where scalar product is maximal of one wavelet
    best_scalar_product = np.dot(wavelet, input_data)
    input_data = input_data-input_data.mean()
    t0 = 0
    for j in range(len(input_data)):
        wavelet_shifted = np.roll(wavelet, j)
        scalar_product = np.dot(wavelet_shifted, input_data)
        if scalar_product>best_scalar_product:
            best_scalar_product = scalar_product
            t0 = j
    print('Best scalar product :' + str(best_scalar_product))
    return t0