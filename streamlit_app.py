"""
Wavelet Decomposition Analysis Interface
==========================================
Interactive Streamlit app for analyzing time series using wavelet decomposition.

Based on the Clerjon & Perdu (2019) methodology.

FINAL VERSION WITH ALL IMPROVEMENTS APPLIED
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import sys
from io import BytesIO

# Import custom modules
from file_manager import WaveletFileManager
from wavelet_decomposition import wavelet_decomposition_single_TS, reconstruct
from plots import plot_betas_heatmap, fft
from import_excel import import_excel
from calc_EPN import calc_epn

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Wavelet Decomposition Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Extended width
# ============================================================================

st.markdown("""
<style>
    /* Extend page width */
    .main .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #A23B72;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE AND DESCRIPTION
# ============================================================================

st.markdown('<div class="main-header">📊 Wavelet Decomposition Analysis</div>', unsafe_allow_html=True)

st.markdown("""
This interactive interface allows you to analyze time series data using wavelet decomposition 
following the methodology of Clerjon & Perdu (2019).

**Workflow:**
1. 📁 Upload your Excel file with time series data
2. 🎯 Select and visualize signal type and year to analyze
3. 🚀 Run wavelet decomposition (15 time scales)
4. 📈 Visualize results (heatmap, FFT spectrum)
5. 🔄 Reconstruct signal with selected time scales
6. 📥 Export workflow as HTML/PDF
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## 🎛️ Configuration")

# Wavelet shape selection
st.sidebar.markdown("### Wavelet Shape")
wavelet_shape = st.sidebar.radio(
    "Select wavelet shape",
    options=['square', 'sine'],
    index=0,
    help="Square wavelets are faster and more commonly used. Sine wavelets provide smoother decomposition."
)

st.sidebar.markdown(f"""
<div class="info-box">
<b>Selected shape:</b> {wavelet_shape}<br>
<small>{'Standard for energy analysis' if wavelet_shape == 'square' else 'Alternative smooth decomposition'}</small>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# STEP 1: FILE UPLOAD AND DATA IMPORT
# ============================================================================

st.markdown('<div class="section-header">📁 Step 1: Upload Data File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Excel file with time series data",
    type=['xlsx', 'xls'],
    help="File should contain columns: 'Consumption', 'Wind', 'PV' with time series data"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Data import parameters
    st.markdown("### Import Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dpd = st.number_input(
            "Data points per day (original)",
            min_value=1,
            max_value=96,
            value=48,
            help="Original sampling rate in the data"
        )
    
    with col2:
        ndpd = st.number_input(
            "Data points per day (interpolated)",
            min_value=1,
            max_value=128,
            value=64,
            help="Target sampling rate after interpolation"
        )
    
    with col3:
        dpy = st.number_input(
            "Days per year",
            min_value=1,
            max_value=366,
            value=365,
            help="Number of days per year (non-leap year)"
        )
    
    # Import data
    if st.button("🔄 Import Data"):
        with st.spinner("Importing time series data..."):
            try:
                # Available time series in the file
                time_series_options = ['Consumption', 'Wind', 'PV']
                
                # Import data
                stacked_input_data, years = import_excel(
                    "",  # path
                    temp_file_path,  # file name
                    dpd,
                    ndpd,
                    dpy,
                    time_series_options,
                    interp=True
                )
                
                # Store in session state
                st.session_state['data_imported'] = True
                st.session_state['stacked_input_data'] = stacked_input_data
                st.session_state['years'] = years
                st.session_state['dpd'] = dpd
                st.session_state['ndpd'] = ndpd
                st.session_state['dpy'] = dpy
                st.session_state['signal_length'] = ndpd * dpy
                st.session_state['wavelet_shape'] = wavelet_shape
                
                st.success("✅ Data imported successfully!")
                
                # Display data info
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"""
                **Data Information:**
                - Available signals: {', '.join(time_series_options)}
                - Years available: {', '.join(years)} ({len(years)} years)
                - Original sampling: {dpd} points/day
                - Interpolated sampling: {ndpd} points/day
                - Days per year: {dpy}
                - Total points per year: {ndpd * dpy:,}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error importing data: {str(e)}")
                st.exception(e)

# ============================================================================
# STEP 2: SIGNAL SELECTION AND DYNAMIC SUBPLOT VISUALIZATION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🎯 Step 2: Select and Visualize Signal</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        signal_type = st.selectbox(
            "Select signal type to analyze",
            options=['PV', 'Wind', 'Consumption'],
            help="Choose which time series to analyze"
        )
    
    with col2:
        year_to_process = st.selectbox(
            "Select year to analyze",
            options=st.session_state['years'],
            help="Choose which year to process"
        )
    
    # Country/Region name
    country_name = st.text_input(
        "Country/Region name",
        value="France",
        help="Name for file organization"
    )
    
    # Store selections in session state
    st.session_state['signal_type'] = signal_type
    st.session_state['year_to_process'] = year_to_process
    st.session_state['country_name'] = country_name
    st.session_state['wavelet_shape'] = wavelet_shape
    
    # ========================================================================
    # DYNAMIC SUBPLOT SYSTEM
    # ========================================================================
    
    st.markdown("### 📊 Time Series Visualization")
    
    # Initialize subplot configuration in session state
    if 'subplots_config' not in st.session_state:
        st.session_state['subplots_config'] = [
            {
                'id': 0,
                'signals': [signal_type],
                'years': [year_to_process],
                'row': 0,
                'col': 0
            }
        ]
    
    # Display current subplots configuration
    st.markdown("#### Configure Subplots")
    
    for idx, subplot_cfg in enumerate(st.session_state['subplots_config']):
        with st.expander(f"Subplot {idx + 1} (Row {subplot_cfg['row'] + 1}, Col {subplot_cfg['col'] + 1})", expanded=(idx == 0)):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                subplot_cfg['signals'] = st.multiselect(
                    "Select signals",
                    options=['Consumption', 'Wind', 'PV'],
                    default=subplot_cfg['signals'],
                    key=f"signals_{idx}"
                )
            
            with col2:
                subplot_cfg['years'] = st.multiselect(
                    "Select years",
                    options=st.session_state['years'],
                    default=subplot_cfg['years'],
                    key=f"years_{idx}"
                )
            
            with col3:
                if idx > 0:  # Can't remove first subplot
                    if st.button("🗑️ Remove", key=f"remove_{idx}"):
                        st.session_state['subplots_config'].pop(idx)
                        st.rerun()
    
    # Add subplot buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Subplot Below"):
            current_rows = max([cfg['row'] for cfg in st.session_state['subplots_config']]) + 1
            
            st.session_state['subplots_config'].append({
                'id': len(st.session_state['subplots_config']),
                'signals': [signal_type],
                'years': [year_to_process],
                'row': current_rows,
                'col': 0
            })
            st.rerun()
    
    with col2:
        if st.button("➕ Add Subplot Right"):
            current_cols = max([cfg['col'] for cfg in st.session_state['subplots_config']]) + 1
            
            st.session_state['subplots_config'].append({
                'id': len(st.session_state['subplots_config']),
                'signals': [signal_type],
                'years': [year_to_process],
                'row': 0,
                'col': current_cols
            })
            st.rerun()
    
    # Plot button
    if st.button("📈 Generate Plots"):
        with st.spinner("Generating plots..."):
            try:
                # Determine grid size
                max_row = max([cfg['row'] for cfg in st.session_state['subplots_config']]) + 1
                max_col = max([cfg['col'] for cfg in st.session_state['subplots_config']]) + 1
                
                # Create subplot titles
                subplot_titles = []
                for r in range(max_row):
                    for c in range(max_col):
                        # Find subplot for this position
                        subplot = next((s for s in st.session_state['subplots_config'] 
                                      if s['row'] == r and s['col'] == c), None)
                        if subplot and subplot['signals'] and subplot['years']:
                            signals_str = ', '.join(subplot['signals'])
                            years_str = ', '.join(subplot['years'])
                            subplot_titles.append(f"{signals_str} - {country_name} ({years_str})")
                        else:
                            subplot_titles.append("")
                
                # Create figure
                fig = make_subplots(
                    rows=max_row,
                    cols=max_col,
                    subplot_titles=subplot_titles,
                    horizontal_spacing=0.08,
                    vertical_spacing=0.12
                )
                
                # Color scheme
                signal_colors = {
                    'Consumption': '#2E86AB',
                    'Wind': '#A23B72',
                    'PV': '#F18F01'
                }
                
                # Line styles for years
                line_styles = ['solid', 'dash', 'dot', 'dashdot']
                
                # Add traces for each subplot
                points_per_year = st.session_state['signal_length']
                time_axis = np.linspace(0, st.session_state['dpy'], points_per_year)
                
                for subplot_cfg in st.session_state['subplots_config']:
                    if not subplot_cfg['signals'] or not subplot_cfg['years']:
                        continue
                        
                    row = subplot_cfg['row'] + 1  # plotly uses 1-indexed
                    col = subplot_cfg['col'] + 1
                    
                    for sig in subplot_cfg['signals']:
                        for year_idx, year in enumerate(subplot_cfg['years']):
                            # Extract data
                            years_available = st.session_state['years']
                            year_index = years_available.index(year)
                            start_idx = year_index * points_per_year
                            end_idx = (year_index + 1) * points_per_year
                            signal_data = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                            
                            # Create legend name with year
                            legend_name = f"{sig} ({year})"
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis,
                                    y=signal_data,
                                    mode='lines',
                                    name=legend_name,
                                    line=dict(
                                        color=signal_colors.get(sig, '#333333'),
                                        width=1.5,
                                        dash=line_styles[year_idx % len(line_styles)]
                                    ),
                                    showlegend=True
                                ),
                                row=row,
                                col=col
                            )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Time (days)", row=row, col=col)
                    fig.update_yaxes(title_text="Normalized Power (MW)", row=row, col=col)
                
                # Update layout
                fig.update_layout(
                    height=400 * max_row,
                    showlegend=True,
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error plotting: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # STEP 3: RUN DECOMPOSITION WITH UPDATED MATH CONTENT
    # ============================================================================
    
    st.markdown('<div class="section-header">🚀 Step 3: Run Wavelet Decomposition</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### About Wavelet Decomposition
    
    This implementation follows the methodology from **Clerjon & Perdu (2019)** published in 
    *Energy & Environmental Science*: *"Matching intermittency and electricity storage characteristics 
    through time scale analysis: an energy return on investment comparison"*.
    
    #### Time Scale Structure
    
    The decomposition uses **15 fixed time scales** optimally distributed across three frequency bands:
    """)
    
    # Create visual display of time scales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🕐 Daily Wavelets (6 scales)**
        - 0.75h (45 min)
        - 1.5h (90 min)
        - 3h
        - 6h
        - 12h (half-day)
        - 24h (day)
        
        *Captures: Intra-day variations, demand peaks, solar cycles*
        """)
    
    with col2:
        st.markdown("""
        **📅 Weekly Wavelets (3 scales)**
        - 42h
        - 84h (3.5 days)
        - 168h (week)
        
        *Captures: Weekly patterns, workday/weekend cycles, weather systems*
        """)
    
    with col3:
        st.markdown("""
        **🌍 Yearly Wavelets (6 scales)**
        - 273.75h (~11 days)
        - 547.5h (~23 days)
        - 1095h (~45 days)
        - 2190h (~91 days)
        - 4380h (~6 months)
        - 8760h (year)
        
        *Captures: Seasonal patterns, annual trends*
        """)
    
    st.markdown("""
    #### Mathematical Foundation
    
    The decomposition uses **square wavelets** by default (or sine wavelets if selected), which are:
    - **Orthogonal**: Each time scale is independent
    - **Additive**: Signal = sum of all components
    - **Complete Dictionary**: The wavelet set constitutes a rich dictionary that enables full reconstruction of the input signal
    
    ##### Example: Haar Wavelet
    
    Below is an illustration of a Haar wavelet (simplest square wavelet):
    
    ```
    ┌─────────────────────────────────┐
    │  Haar Wavelet Example           │
    │                                 │
    │    +1 ┌─────┐                  │
    │       │     │                  │
    │     0 ┴─────┴─────             │
    │                   │            │
    │   -1              └─────┐      │
    │                         │      │
    │       └─────────────────┘      │
    │                                │
    │   ◄────── Period ─────►        │
    │                                │
    │   Positive half: +1            │
    │   Negative half: -1            │
    │   Mean: 0                      │
    └─────────────────────────────────┘
    ```
    
    The Haar wavelet alternates between +1 and -1 over its support, with zero mean.
    This simple shape enables detection of transitions and changes in the signal.
    
    #### Decomposition Process
    
    1. **Translation Optimization** (optional)
       - Finds optimal circular shift for each wavelet
       - Maximizes correlation with signal
       - Cached for reuse
    
    2. **Matrix Generation**
       - Creates sparse wavelet transformation matrix
       - Dimensions: (signal_length × number_of_wavelets)
       - Stored as compressed .npz file
    
    3. **Coefficient Calculation**
       - Solves: Signal = Matrix × Betas
       - Uses least-squares solver (LSQR algorithm)
       - Betas represent amplitude at each time scale
    
    4. **Result Storage**
       - Files saved in: `results/{region}/{shape}/`
       - Enables fast reanalysis without recomputation
    """)
    
    # Fixed parameters display
    vy = 6  # Yearly wavelets
    vw = 3  # Weekly wavelets
    vd = 6  # Daily wavelets
    
    time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 
                   273.75, 547.5, 1095., 2190., 4380., 8760.]
    
    st.markdown(f"""
    <div class="info-box">
    <b>Current Configuration:</b><br>
    - Yearly wavelets: {vy} (scales: 273.75h - 8760h)<br>
    - Weekly wavelets: {vw} (scales: 42h - 168h)<br>
    - Daily wavelets: {vd} (scales: 0.75h - 24h)<br>
    - Total time scales: {len(time_scales)}<br>
    - Wavelet shape: <b>{wavelet_shape}</b><br>
    - Signal resolution: {st.session_state['ndpd']} points/day<br>
    - Total data points: {st.session_state['signal_length']:,}
    </div>
    """, unsafe_allow_html=True)
    
    recompute_translation = st.checkbox(
        "Recompute translations",
        value=False,
        help="If checked, recompute optimal translations (slower, ~1-2 min extra). Otherwise, load cached translations (<5 sec)."
    )
    
    if st.button("🚀 Run Wavelet Decomposition", type="primary"):
        
        # Create containers for progress display
        progress_container = st.container()
        log_container = st.container()
        
        with st.spinner(f"Running wavelet decomposition for {signal_type} signal in {year_to_process}..."):
            try:
                # Capture print outputs to display progress
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                # Extract single year data
                years_available = st.session_state['years']
                year_index = years_available.index(year_to_process)
                points_per_year = st.session_state['signal_length']
                start_idx = year_index * points_per_year
                end_idx = (year_index + 1) * points_per_year
                
                TS_single_year = st.session_state['stacked_input_data'][signal_type][start_idx:end_idx]
                
                # Run decomposition
                trans_file, matrix_files, results_betas, trans = wavelet_decomposition_single_TS(
                    TS_single_year,
                    year=year_to_process,
                    multi_year=None,
                    country_name=country_name,
                    signal_type=signal_type,
                    wl_shape=wavelet_shape,
                    recompute_translation=recompute_translation,
                    dpd=st.session_state['dpd'],
                    ndpd=st.session_state['ndpd'],
                    vy=vy,
                    vw=vw,
                    vd=vd
                )
                
                # Restore stdout and get captured output
                sys.stdout = old_stdout
                output = buffer.getvalue()
                
                # Display captured progress log
                if output:
                    with log_container:
                        with st.expander("📋 View Decomposition Log", expanded=True):
                            st.code(output, language="text")
                
                # Store results in session state
                st.session_state['decomposition_done'] = True
                st.session_state['trans_file'] = trans_file
                st.session_state['matrix_files'] = matrix_files
                st.session_state['results_betas'] = results_betas
                st.session_state['trans'] = trans
                st.session_state['vy'] = vy
                st.session_state['vw'] = vw
                st.session_state['vd'] = vd
                st.session_state['time_scales'] = time_scales
                
                # Success message with details
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Decomposition Complete!</b><br><br>
                <b>Output Files:</b><br>
                • Translation: <code>{trans_file}</code><br>
                • Matrix: <code>{matrix_files[0]}</code><br>
                • Coefficients: Betas computed for {year_to_process}<br><br>
                <b>Results:</b><br>
                • {len(time_scales)} time scales analyzed<br>
                • {len(results_betas[year_to_process])} coefficient sets<br>
                • Ready for visualization and reconstruction
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                # Restore stdout in case of error
                sys.stdout = old_stdout
                st.error(f"❌ Error during decomposition: {str(e)}")
                st.exception(e)
                
                # Show partial log if available
                if 'buffer' in locals():
                    output = buffer.getvalue()
                    if output:
                        with st.expander("📋 Partial Log (before error)"):
                            st.code(output, language="text")

# ============================================================================
# STEP 4: VISUALIZATION WITH AUTOMATIC DECOMPOSITION
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">📈 Step 4: Visualization</div>', unsafe_allow_html=True)

  
# Initialize storage for ALL decomposition results (by signal and year)
if 'all_decompositions' not in st.session_state:
    st.session_state['all_decompositions'] = {}

    # Define initial_key for the current signal/year combination
    # Use .get() to safely access session state values
    signal_type_current = st.session_state.get('signal_type', 'Unknown')
    year_current = st.session_state.get('year_to_process', 'Unknown')
    initial_key = f"{signal_type_current}_{year_current}"

    # Store the initial decomposition result (only if we have valid results)
    if initial_key not in st.session_state['all_decompositions'] and 'results_betas' in st.session_state:
        st.session_state['all_decompositions'][initial_key] = {
            'results_betas': st.session_state['results_betas'],
            'trans_file': st.session_state.get('trans_file'),
            'matrix_files': st.session_state.get('matrix_files'),
            'trans': st.session_state.get('trans')
        }
    
    # Initialize storage for generated plots
    if 'generated_plots' not in st.session_state:
        st.session_state['generated_plots'] = []
    
    # Initialize storage for plot configurations
    if 'plot_configs' not in st.session_state:
        st.session_state['plot_configs'] = []
    
    st.markdown("""
    Create and arrange multiple visualizations. If you select a signal that hasn't been 
    decomposed yet, it will be automatically decomposed before visualization.
    """)
    
    # Show which signals have been decomposed
    with st.expander("ℹ️ Decomposition Status", expanded=False):
        st.markdown("**Currently decomposed signal combinations:**")
        if st.session_state['all_decompositions']:
            for key in st.session_state['all_decompositions'].keys():
                signal, year = key.split('_')
                st.write(f"✅ {signal} - {year}")
        else:
            st.write("No decompositions yet")
    
    # ========================================================================
    # HELPER FUNCTION: Ensure Signal is Decomposed
    # ========================================================================
    @st.cache_data
    def ensure_decomposition(signal, year):
        """
        Check if decomposition exists for this signal/year.
        If not, run decomposition automatically.
        Returns: (success, results_betas or error_message)
        """
        key = f"{signal}_{year}"
        
        # Check if already decomposed
        if key in st.session_state['all_decompositions']:
            return True, st.session_state['all_decompositions'][key]['results_betas']
        
        # Need to decompose this signal/year
        st.info(f"⏳ Decomposing {signal} for {year}... This may take a moment.")
        
        try:
            # Extract signal data for this year
            years_available = st.session_state['years']
            year_index = years_available.index(year)
            points_per_year = st.session_state['signal_length']
            start_idx = year_index * points_per_year
            end_idx = (year_index + 1) * points_per_year
            
            TS_single_year = st.session_state['stacked_input_data'][signal][start_idx:end_idx]
            
            # Run decomposition
            trans_file, matrix_files, results_betas, trans = wavelet_decomposition_single_TS(
                TS_single_year,
                year=year,
                multi_year=None,
                country_name=st.session_state['country_name'],
                signal_type=signal,
                wl_shape=st.session_state['wavelet_shape'],
                recompute_translation=False,  # Use cached translations
                dpd=st.session_state['dpd'],
                ndpd=st.session_state['ndpd'],
                vy=st.session_state['vy'],
                vw=st.session_state['vw'],
                vd=st.session_state['vd']
            )
            
            # Store results
            st.session_state['all_decompositions'][key] = {
                'results_betas': results_betas,
                'trans_file': trans_file,
                'matrix_files': matrix_files,
                'trans': trans
            }
            
            st.success(f"✅ {signal} ({year}) decomposed successfully!")
            return True, results_betas
            
        except Exception as e:
            error_msg = f"Error decomposing {signal} ({year}): {str(e)}"
            st.error(error_msg)
            return False, error_msg
    
    # ========================================================================
    # SECTION 1: ADD NEW PLOT CONFIGURATION
    # ========================================================================
    
    st.markdown("### Add New Visualization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Plot", use_container_width=True):
            st.session_state['plot_configs'].append({
                'id': len(st.session_state['plot_configs']),
                'type': 'heatmap',
                'year': st.session_state['year_to_process'],
                'signal': st.session_state['signal_type'],
                'time_scales': list(st.session_state['time_scales']),
                'generated': False,
                'fig': None
            })
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Plots", use_container_width=True):
            st.session_state['generated_plots'] = []
            st.session_state['plot_configs'] = []
            st.rerun()
    
    with col3:
        if st.session_state['generated_plots']:
            st.metric("Generated Plots", len(st.session_state['generated_plots']))
    
    # ========================================================================
    # SECTION 2: CONFIGURE PENDING PLOTS
    # ========================================================================
    
    pending_plots = [p for p in st.session_state['plot_configs'] if not p['generated']]
    
    if pending_plots:
        st.markdown("---")
        st.markdown("### Configure New Plots")
        
        for idx, plot_cfg in enumerate(pending_plots):
            # Find the index in the full list
            full_idx = st.session_state['plot_configs'].index(plot_cfg)
            
            with st.expander(f"⚙️ Plot {full_idx + 1} - Configuration", expanded=True):
                
                # Configuration columns
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    plot_cfg['type'] = st.selectbox(
                        "Plot type",
                        options=['heatmap', 'fft'],
                        index=0 if plot_cfg['type'] == 'heatmap' else 1,
                        key=f"type_config_{full_idx}"
                    )
                
                with col2:
                    plot_cfg['year'] = st.selectbox(
                        "Year",
                        options=st.session_state['years'],
                        index=st.session_state['years'].index(plot_cfg['year']),
                        key=f"year_config_{full_idx}"
                    )
                
                with col3:
                    plot_cfg['signal'] = st.selectbox(
                        "Signal type",
                        options=['Consumption', 'Wind', 'PV'],
                        index=['Consumption', 'Wind', 'PV'].index(plot_cfg['signal']),
                        key=f"signal_config_{full_idx}"
                    )
                
                with col4:
                    if st.button("❌", key=f"cancel_{full_idx}", help="Cancel this plot"):
                        st.session_state['plot_configs'].remove(plot_cfg)
                        st.rerun()
                
                # Check if this signal/year has been decomposed
                decomp_key = f"{plot_cfg['signal']}_{plot_cfg['year']}"
                is_decomposed = decomp_key in st.session_state['all_decompositions']
                
                if not is_decomposed:
                    st.warning(f"⚠️ {plot_cfg['signal']} ({plot_cfg['year']}) has not been decomposed yet. "
                             f"It will be automatically decomposed when you generate this plot.")
                
                # Time scale selection for heatmap
                if plot_cfg['type'] == 'heatmap':
                    st.markdown("**Select Time Scales:**")
                    
                    # Initialize checkbox state for this plot
                    checkbox_key = f'plot_checks_{full_idx}'
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = {ts: True for ts in st.session_state['time_scales']}
                    
                    # Select/Deselect buttons
                    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 3])
                    
                    with col_btn1:
                        if st.button("✅ All", key=f"sel_all_config_{full_idx}"):
                            for ts in st.session_state['time_scales']:
                                st.session_state[checkbox_key][ts] = True
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("❌ None", key=f"desel_all_config_{full_idx}"):
                            for ts in st.session_state['time_scales']:
                                st.session_state[checkbox_key][ts] = False
                            st.rerun()
                    
                    # Checkboxes in single line
                    checkbox_cols = st.columns(15)
                    
                    time_scale_info = {
                        0.75: "0.75h", 1.5: "1.5h", 3.0: "3h", 6.0: "6h", 12.0: "12h", 24.0: "24h",
                        42.0: "42h", 84.0: "84h", 168.0: "168h", 273.75: "273.75h", 547.5: "547.5h",
                        1095.0: "1095h", 2190.0: "2190h", 4380.0: "4380h", 8760.0: "8760h"
                    }
                    
                    for i, ts in enumerate(st.session_state['time_scales']):
                        with checkbox_cols[i]:
                            current = st.session_state[checkbox_key].get(ts, True)
                            new = st.checkbox(
                                time_scale_info[ts],
                                value=current,
                                key=f"check_config_{full_idx}_{ts}"
                            )
                            if new != current:
                                st.session_state[checkbox_key][ts] = new
                    
                    # Update plot config with selected scales
                    plot_cfg['time_scales'] = [
                        ts for ts in st.session_state['time_scales']
                        if st.session_state[checkbox_key].get(ts, True)
                    ]
                    
                    # Show selection count
                    if plot_cfg['time_scales']:
                        st.info(f"✅ Selected {len(plot_cfg['time_scales'])} time scales")
                    else:
                        st.warning("⚠️ No time scales selected")
                
                # Generate button
                st.markdown("---")
                col_gen1, col_gen2, col_gen3 = st.columns([1, 2, 1])
                
                with col_gen2:
                    if st.button(
                        f"📊 Generate Plot {full_idx + 1}",
                        key=f"generate_{full_idx}",
                        type="primary",
                        use_container_width=True
                    ):
                        if plot_cfg['type'] == 'heatmap' and not plot_cfg['time_scales']:
                            st.error("⚠️ Please select at least one time scale for heatmap")
                        else:
                            with st.spinner(f"Generating {plot_cfg['type']}..."):
                                try:
                                    # CRITICAL FIX: Ensure this signal/year is decomposed
                                    success, results_or_error = ensure_decomposition(
                                        plot_cfg['signal'], 
                                        plot_cfg['year']
                                    )
                                    
                                    if not success:
                                        st.error(f"Cannot generate plot: {results_or_error}")
                                        continue
                                    
                                    results_betas = results_or_error
                                    
                                    if plot_cfg['type'] == 'heatmap':
                                        # Generate heatmap using CORRECT betas
                                        fig = plot_betas_heatmap(
                                            results_betas=results_betas,  # Use correct betas!
                                            country_name=st.session_state['country_name'],
                                            signal_type=plot_cfg['signal'],
                                            vy=st.session_state['vy'],
                                            vw=st.session_state['vw'],
                                            vd=st.session_state['vd'],
                                            ndpd=st.session_state['ndpd'],
                                            dpy=st.session_state['dpy'],
                                            year=plot_cfg['year'],
                                            years=[plot_cfg['year']],
                                            time_scales=st.session_state['time_scales'],
                                            reconstructed_time_scales=plot_cfg['time_scales'],
                                            cmin=-0.1,
                                            cmax=0.1,
                                            ccenter=None,
                                            wl_shape=st.session_state['wavelet_shape']
                                        )
                                    
                                    elif plot_cfg['type'] == 'fft':
                                        # Get data for selected year and signal
                                        years_available = st.session_state['years']
                                        year_index = years_available.index(plot_cfg['year'])
                                        points_per_year = st.session_state['signal_length']
                                        start_idx = year_index * points_per_year
                                        end_idx = (year_index + 1) * points_per_year
                                        
                                        input_data = st.session_state['stacked_input_data'][plot_cfg['signal']][start_idx:end_idx]
                                        
                                        fig = fft(
                                            ndpd=st.session_state['ndpd'],
                                            dpy=st.session_state['dpy'],
                                            signal_type=plot_cfg['signal'],
                                            year=plot_cfg['year'],
                                            input_data=input_data
                                        )
                                    
                                    # Store the generated plot
                                    plot_cfg['fig'] = fig
                                    plot_cfg['generated'] = True
                                    plot_cfg['title'] = f"{plot_cfg['type'].upper()}: {plot_cfg['signal']} - {plot_cfg['year']}"
                                    
                                    # Add to generated plots
                                    st.session_state['generated_plots'].append(plot_cfg)
                                    
                                    st.success(f"✅ Plot {full_idx + 1} generated!")
                                    st.rerun()
                                
                                except Exception as e:
                                    st.error(f"❌ Error generating plot: {str(e)}")
                                    st.exception(e)
    
    # ========================================================================
    # SECTION 3: DISPLAY ALL GENERATED PLOTS
    # ========================================================================
    
    if st.session_state['generated_plots']:
        st.markdown("---")
        st.markdown("### Generated Visualizations")
        
        # Layout options
        col_layout1, col_layout2, col_layout3 = st.columns(3)
        
        with col_layout1:
            layout_option = st.radio(
                "Layout",
                options=["Grid (2 columns)", "Grid (3 columns)", "Stacked (1 column)"],
                index=0,
                help="Choose how to arrange multiple plots"
            )
        
        with col_layout2:
            if st.button("🔄 Refresh All Plots", use_container_width=True):
                st.rerun()
        
        with col_layout3:
            st.write(f"**{len(st.session_state['generated_plots'])} plot(s) displayed**")
        
        st.markdown("---")
        
        # Determine number of columns
        if layout_option == "Grid (2 columns)":
            n_cols = 2
        elif layout_option == "Grid (3 columns)":
            n_cols = 3
        else:
            n_cols = 1
        
        # Display plots in grid
        plots = st.session_state['generated_plots']
        
        for i in range(0, len(plots), n_cols):
            cols = st.columns(n_cols)
            
            for j in range(n_cols):
                idx = i + j
                if idx < len(plots):
                    plot_data = plots[idx]
                    
                    with cols[j]:
                        # Plot header with remove button
                        col_title, col_remove = st.columns([4, 1])
                        
                        with col_title:
                            st.markdown(f"**{plot_data['title']}**")
                        
                        with col_remove:
                            if st.button("🗑️", key=f"remove_plot_{idx}", help="Remove this plot"):
                                st.session_state['generated_plots'].pop(idx)
                                # Also remove from configs
                                if plot_data in st.session_state['plot_configs']:
                                    st.session_state['plot_configs'].remove(plot_data)
                                st.rerun()
                        
                        # Display the plot
                        if plot_data['fig'] is not None:
                            st.pyplot(plot_data['fig'])
                            plt.close(plot_data['fig']) 
                            
                            # Show plot info
                            with st.expander(f"ℹ️ Plot Info", expanded=False):
                                st.write(f"**Type:** {plot_data['type']}")
                                st.write(f"**Signal:** {plot_data['signal']}")
                                st.write(f"**Year:** {plot_data['year']}")
                                if plot_data['type'] == 'heatmap':
                                    st.write(f"**Time scales:** {len(plot_data['time_scales'])} selected")
                        else:
                            st.error("Plot data not available")
# ============================================================================
# STEP 5: SIGNAL RECONSTRUCTION - COMPLETE FIXED VERSION
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">🔄 Step 5: Signal Reconstruction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Reconstruct signals using selected time scales. You can reconstruct different signals 
    and years by selecting them below. If a signal hasn't been decomposed, it will be 
    automatically decomposed before reconstruction.
    """)
    
    # Initialize storage for generated reconstructions
    if 'generated_reconstructions' not in st.session_state:
        st.session_state['generated_reconstructions'] = []
    
    # Initialize storage for reconstruction configurations
    if 'recon_configs' not in st.session_state:
        st.session_state['recon_configs'] = []
    
    # ========================================================================
    # SECTION 1: ADD NEW RECONSTRUCTION CONFIGURATION
    # ========================================================================
    
    st.markdown("### Add New Reconstruction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Reconstruction", use_container_width=True):
            st.session_state['recon_configs'].append({
                'id': len(st.session_state['recon_configs']),
                'signal': st.session_state['signal_type'],
                'year': st.session_state['year_to_process'],
                'time_scales': list(st.session_state['time_scales']),
                'add_offset': False,
                'generated': False,
                'fig': None,
                'reconstructed_signal': None
            })
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Reconstructions", use_container_width=True):
            st.session_state['generated_reconstructions'] = []
            st.session_state['recon_configs'] = []
            st.rerun()
    
    with col3:
        if st.session_state['generated_reconstructions']:
            st.metric("Generated", len(st.session_state['generated_reconstructions']))
    
    # ========================================================================
    # HELPER FUNCTION: Ensure Signal is Decomposed (from Step 4)
    # ========================================================================
    
    def ensure_decomposition_for_reconstruction(signal, year):
        """
        Check if decomposition exists for this signal/year.
        If not, run decomposition automatically.
        Returns: (success, results_betas or error_message)
        """
        key = f"{signal}_{year}"
        
        # Check if already decomposed
        if key in st.session_state.get('all_decompositions', {}):
            return True, st.session_state['all_decompositions'][key]['results_betas']
        
        # Need to decompose this signal/year
        st.info(f"⏳ Decomposing {signal} for {year}... This may take a moment.")
        
        try:
            # Extract signal data for this year
            years_available = st.session_state['years']
            year_index = years_available.index(year)
            points_per_year = st.session_state['signal_length']
            start_idx = year_index * points_per_year
            end_idx = (year_index + 1) * points_per_year
            
            TS_single_year = st.session_state['stacked_input_data'][signal][start_idx:end_idx]
            
            # Run decomposition
            trans_file, matrix_files, results_betas, trans = wavelet_decomposition_single_TS(
                TS_single_year,
                year=year,
                multi_year=None,
                country_name=st.session_state['country_name'],
                signal_type=signal,
                wl_shape=st.session_state['wavelet_shape'],
                recompute_translation=False,  # Use cached translations
                dpd=st.session_state['dpd'],
                ndpd=st.session_state['ndpd'],
                vy=st.session_state['vy'],
                vw=st.session_state['vw'],
                vd=st.session_state['vd']
            )
            
            # Initialize all_decompositions if not exists
            if 'all_decompositions' not in st.session_state:
                st.session_state['all_decompositions'] = {}
            
            # Store results
            st.session_state['all_decompositions'][key] = {
                'results_betas': results_betas,
                'trans_file': trans_file,
                'matrix_files': matrix_files,
                'trans': trans
            }
            
            st.success(f"✅ {signal} ({year}) decomposed successfully!")
            return True, results_betas
            
        except Exception as e:
            error_msg = f"Error decomposing {signal} ({year}): {str(e)}"
            st.error(error_msg)
            return False, error_msg
    
    # ========================================================================
    # SECTION 2: CONFIGURE PENDING RECONSTRUCTIONS
    # ========================================================================
    
    pending_recons = [r for r in st.session_state['recon_configs'] if not r['generated']]
    
    if pending_recons:
        st.markdown("---")
        st.markdown("### Configure New Reconstructions")
        
        for idx, recon_cfg in enumerate(pending_recons):
            # Find the index in the full list
            full_idx = st.session_state['recon_configs'].index(recon_cfg)
            
            with st.expander(f"⚙️ Reconstruction {full_idx + 1} - Configuration", expanded=True):
                
                # Signal and Year selection
                st.markdown("**Select Signal and Year:**")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    recon_cfg['signal'] = st.selectbox(
                        "Signal type",
                        options=['Consumption', 'Wind', 'PV'],
                        index=['Consumption', 'Wind', 'PV'].index(recon_cfg['signal']),
                        key=f"signal_recon_{full_idx}"
                    )
                
                with col2:
                    recon_cfg['year'] = st.selectbox(
                        "Year",
                        options=st.session_state['years'],
                        index=st.session_state['years'].index(recon_cfg['year']),
                        key=f"year_recon_{full_idx}"
                    )
                
                with col3:
                    if st.button("❌", key=f"cancel_recon_{full_idx}", help="Cancel"):
                        st.session_state['recon_configs'].remove(recon_cfg)
                        st.rerun()
                
                # Check if this signal/year has been decomposed
                decomp_key = f"{recon_cfg['signal']}_{recon_cfg['year']}"
                is_decomposed = decomp_key in st.session_state.get('all_decompositions', {})
                
                if not is_decomposed:
                    st.warning(f"⚠️ {recon_cfg['signal']} ({recon_cfg['year']}) has not been decomposed yet. "
                             f"It will be automatically decomposed when you generate this reconstruction.")
                
                # Time scale selection
                st.markdown("**Select Time Scales:**")
                
                # Initialize checkbox state for this reconstruction
                checkbox_key = f'recon_checks_{full_idx}'
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = {ts: True for ts in st.session_state['time_scales']}
                
                # Select/Deselect buttons - FIXED VERSION
                col_btn1, col_btn2, col_spacer = st.columns([1, 1, 3])
                
                with col_btn1:
                    if st.button("✅ All", key=f"sel_all_recon_{full_idx}"):
                        for ts in st.session_state['time_scales']:
                            st.session_state[checkbox_key][ts] = True
                        st.rerun()
                
                with col_btn2:
                    if st.button("❌ None", key=f"desel_all_recon_{full_idx}"):
                        for ts in st.session_state['time_scales']:
                            st.session_state[checkbox_key][ts] = False
                        st.rerun()
                
                # Checkboxes in single line - FIXED VERSION
                checkbox_cols = st.columns(15)
                
                time_scale_info = {
                    0.75: "0.75h", 1.5: "1.5h", 3.0: "3h", 6.0: "6h", 12.0: "12h", 24.0: "24h",
                    42.0: "42h", 84.0: "84h", 168.0: "168h", 273.75: "273.75h", 547.5: "547.5h",
                    1095.0: "1095h", 2190.0: "2190h", 4380.0: "4380h", 8760.0: "8760h"
                }
                
                for i, ts in enumerate(st.session_state['time_scales']):
                    with checkbox_cols[i]:
                        # CRITICAL FIX: Read first, create, then update if changed
                        current = st.session_state[checkbox_key].get(ts, True)
                        new = st.checkbox(
                            time_scale_info[ts],
                            value=current,
                            key=f"check_recon_{full_idx}_{ts}"
                        )
                        if new != current:
                            st.session_state[checkbox_key][ts] = new
                
                # Update recon config with selected scales
                recon_cfg['time_scales'] = [
                    ts for ts in st.session_state['time_scales']
                    if st.session_state[checkbox_key].get(ts, True)
                ]
                
                # Show selection count
                if recon_cfg['time_scales']:
                    st.info(f"✅ Selected {len(recon_cfg['time_scales'])} time scales")
                else:
                    st.warning("⚠️ No time scales selected")
                
                # Add offset option
                st.markdown("**Options:**")
                recon_cfg['add_offset'] = st.checkbox(
                    "Add offset (DC component)",
                    value=recon_cfg['add_offset'],
                    key=f"offset_recon_{full_idx}"
                )
                
                # Generate button
                st.markdown("---")
                col_gen1, col_gen2, col_gen3 = st.columns([1, 2, 1])
                
                with col_gen2:
                    if st.button(
                        f"🔄 Generate Reconstruction {full_idx + 1}",
                        key=f"generate_recon_{full_idx}",
                        type="primary",
                        use_container_width=True
                    ):
                        if not recon_cfg['time_scales']:
                            st.error("⚠️ Please select at least one time scale")
                        else:
                            with st.spinner(f"Reconstructing {recon_cfg['signal']} for {recon_cfg['year']}..."):
                                try:
                                    # CRITICAL: Ensure this signal/year is decomposed
                                    success, results_or_error = ensure_decomposition_for_reconstruction(
                                        recon_cfg['signal'], 
                                        recon_cfg['year']
                                    )
                                    
                                    if not success:
                                        st.error(f"Cannot reconstruct: {results_or_error}")
                                        continue
                                    
                                    results_betas = results_or_error
                                    
                                    # Load matrix for this signal/year
                                    file_mgr = WaveletFileManager(
                                        region=st.session_state['country_name'],
                                        wl_shape=st.session_state['wavelet_shape']
                                    )
                                    matrix_file = file_mgr.get_matrix_path(recon_cfg['year'])
                                    matrix = sparse.load_npz(matrix_file)
                                    
                                    # Reconstruct using CORRECT betas for this signal/year
                                    reconstructed_signal = reconstruct(
                                        time_scales=st.session_state['time_scales'],
                                        reconstructed_time_scales=recon_cfg['time_scales'],
                                        matrix=matrix,
                                        beta_sheet=results_betas[recon_cfg['year']],
                                        title=f'Reconstruction {full_idx + 1}',
                                        xmin=0,
                                        xmax=st.session_state['dpy'],
                                        dpy=st.session_state['dpy'],
                                        dpd=st.session_state['ndpd'],
                                        add_offset=recon_cfg['add_offset'],
                                        plot=False
                                    )
                                    
                                    # Validate
                                    if reconstructed_signal is None or len(reconstructed_signal) == 0:
                                        st.error("❌ Reconstruction failed")
                                    else:
                                        # Create plot with Plotly
                                        time_axis = np.linspace(0, st.session_state['dpy'], len(reconstructed_signal))
                                        
                                        fig = go.Figure()
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=time_axis,
                                                y=reconstructed_signal,
                                                mode='lines',
                                                name='Reconstructed',
                                                line=dict(color='#2E86AB', width=1.5)
                                            )
                                        )
                                        
                                        # IMPROVED TITLE: Shows signal, year, and time scales
                                        scales_str = ', '.join([f"{ts}h" for ts in recon_cfg['time_scales'][:5]])
                                        if len(recon_cfg['time_scales']) > 5:
                                            scales_str += f" ... (+{len(recon_cfg['time_scales'])-5} more)"
                                        
                                        title_text = (f"{recon_cfg['signal']} - {recon_cfg['year']} - "
                                                     f"{len(recon_cfg['time_scales'])} scales: {scales_str}")
                                        
                                        fig.update_layout(
                                            title=title_text,
                                            xaxis_title='Time (days)',
                                            yaxis_title='Amplitude',
                                            height=400,
                                            template='plotly_white',
                                            hovermode='x unified'
                                        )
                                        
                                        # Store the reconstruction
                                        recon_cfg['fig'] = fig
                                        recon_cfg['reconstructed_signal'] = reconstructed_signal
                                        recon_cfg['generated'] = True
                                        recon_cfg['title'] = title_text
                                        
                                        # Add to generated reconstructions
                                        st.session_state['generated_reconstructions'].append(recon_cfg)
                                        
                                        st.success(f"✅ Reconstruction {full_idx + 1} generated!")
                                        st.rerun()
                                
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                                    st.exception(e)
    
    # ========================================================================
    # SECTION 3: DISPLAY ALL GENERATED RECONSTRUCTIONS
    # ========================================================================
    
    if st.session_state['generated_reconstructions']:
        st.markdown("---")
        st.markdown("### Generated Reconstructions")
        
        # Layout options
        col_layout1, col_layout2, col_layout3 = st.columns(3)
        
        with col_layout1:
            layout_option = st.radio(
                "Layout",
                options=["Grid (2 columns)", "Grid (3 columns)", "Stacked (1 column)"],
                index=0,
                key="recon_layout",
                help="Choose how to arrange multiple reconstructions"
            )
        
        with col_layout2:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_recons"):
                st.rerun()
        
        with col_layout3:
            st.write(f"**{len(st.session_state['generated_reconstructions'])} reconstruction(s)**")
        
        st.markdown("---")
        
        # Determine number of columns
        if layout_option == "Grid (2 columns)":
            n_cols = 2
        elif layout_option == "Grid (3 columns)":
            n_cols = 3
        else:
            n_cols = 1
        
        # Display reconstructions in grid
        recons = st.session_state['generated_reconstructions']
        
        for i in range(0, len(recons), n_cols):
            cols = st.columns(n_cols)
            
            for j in range(n_cols):
                idx = i + j
                if idx < len(recons):
                    recon_data = recons[idx]
                    
                    with cols[j]:
                        # Header with remove button
                        col_title, col_remove = st.columns([4, 1])
                        
                        with col_title:
                            st.markdown(f"**Reconstruction {idx + 1}**")
                            st.caption(f"{recon_data['signal']} - {recon_data['year']}")
                        
                        with col_remove:
                            if st.button("🗑️", key=f"remove_recon_display_{idx}", help="Remove"):
                                st.session_state['generated_reconstructions'].pop(idx)
                                if recon_data in st.session_state['recon_configs']:
                                    st.session_state['recon_configs'].remove(recon_data)
                                st.rerun()
                        
                        # Display the plot with UNIQUE KEY - CRITICAL FIX
                        if recon_data['fig'] is not None:
                            st.plotly_chart(
                                recon_data['fig'], 
                                use_container_width=True,
                                key=f"plotly_recon_{idx}_{recon_data['signal']}_{recon_data['year']}"
                            )
                            
                            # Show info (NOT statistics)
                            with st.expander(f"ℹ️ Info", expanded=False):
                                st.write(f"**Signal:** {recon_data['signal']}")
                                st.write(f"**Year:** {recon_data['year']}")
                                st.write(f"**Time scales:** {len(recon_data['time_scales'])}")
                                st.write(f"**Scales used:** {', '.join([f'{ts}h' for ts in recon_data['time_scales']])}")
                                st.write(f"**Offset:** {'Yes' if recon_data['add_offset'] else 'No'}")
                        else:
                            st.error("Reconstruction data not available")

# ============================================================================
# STEP 6: EPN ANALYSIS - ENERGY FLEXIBILITY REQUIREMENTS
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">⚡ Step 6: EPN Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze energy storage flexibility requirements for different renewable energy mix scenarios.
    Based on **Clerjon & Perdu (2019)** methodology.
    """)
    
    # ========================================================================
    # 6.1 CONFIGURATION
    # ========================================================================
    
    with st.expander("⚙️ EPN Configuration", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            epn_satisfaction_rate = st.slider(
                "Satisfaction Rate (%)",
                min_value=80.0,
                max_value=100.0,
                value=95.0,
                step=0.5,
                help="Percentage of time the load will be met by storage",
                key="epn_satisfaction_slider"
            )
            epn_satisfactions = [epn_satisfaction_rate]
        
        with col2:
            epn_load_factor = st.number_input(
                "Load Factor (MW)",
                min_value=1000,
                max_value=100000,
                value=54000,
                step=1000,
                help="Average power consumption (e.g., 54000 MW for France)",
                key="epn_load_factor_input"
            )
        
        st.markdown("**Select metrics to display:**")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            epn_show_energy = st.checkbox("Energy", value=True, key="epn_show_energy")
        with metric_col2:
            epn_show_uf = st.checkbox("Utilization Factor", value=False, key="epn_show_uf")
        with metric_col3:
            epn_show_service = st.checkbox("Service", value=False, key="epn_show_service")
    
    # ========================================================================
    # 6.2 SCENARIO DEFINITION
    # ========================================================================
    
    st.markdown("### Define Energy Mix Scenarios")
    
    if 'epn_scenarios_config' not in st.session_state:
        st.session_state['epn_scenarios_config'] = [
            {'name': '100% PV', 'pv_share': 1.0},
            {'name': '100% Wind', 'pv_share': 0.0},
        ]
    
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("➕ Add 50/50 Mix", key="epn_add_5050"):
            st.session_state['epn_scenarios_config'].append(
                {'name': '50% PV + 50% Wind', 'pv_share': 0.5}
            )
            st.rerun()
    with btn_col2:
        if st.button("➕ Add Custom", key="epn_add_custom"):
            st.session_state['epn_scenarios_config'].append(
                {'name': 'Custom Mix', 'pv_share': 0.3}
            )
            st.rerun()
    with btn_col3:
        if st.button("🔄 Reset Scenarios", key="epn_reset"):
            st.session_state['epn_scenarios_config'] = [
                {'name': '100% PV', 'pv_share': 1.0},
                {'name': '100% Wind', 'pv_share': 0.0},
            ]
            st.rerun()
    
    # Display and edit scenarios
    epn_scenarios_to_remove = []
    for idx, scenario in enumerate(st.session_state['epn_scenarios_config']):
        scen_col1, scen_col2, scen_col3 = st.columns([2, 3, 1])
        with scen_col1:
            new_name = st.text_input(
                f"Scenario {idx+1} Name", 
                value=scenario['name'],
                key=f"epn_scenario_name_{idx}",
                label_visibility="collapsed"
            )
            st.session_state['epn_scenarios_config'][idx]['name'] = new_name
        with scen_col2:
            new_pv = st.slider(
                f"PV Share for scenario {idx+1}",
                min_value=0.0,
                max_value=1.0,
                value=float(scenario['pv_share']),
                format="%.0f%% PV",
                key=f"epn_scenario_pv_{idx}",
                label_visibility="collapsed"
            )
            st.session_state['epn_scenarios_config'][idx]['pv_share'] = new_pv
        with scen_col3:
            if len(st.session_state['epn_scenarios_config']) > 1:
                if st.button("🗑️", key=f"epn_remove_scenario_{idx}"):
                    epn_scenarios_to_remove.append(idx)
    
    for idx in sorted(epn_scenarios_to_remove, reverse=True):
        st.session_state['epn_scenarios_config'].pop(idx)
        st.rerun()
    
    # ========================================================================
    # 6.3 RUN EPN ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    
    if st.button("🚀 Run EPN Analysis", type="primary", use_container_width=True, key="epn_run_button"):
        
        epn_year = st.session_state['year_to_process']
        epn_time_scales = st.session_state['time_scales']
        epn_dpy = st.session_state.get('dpy', 365)
        
        epn_progress = st.empty()
        
        with st.spinner("Running EPN Analysis..."):
            
            # ----------------------------------------------------------------
            # STEP A: Ensure all signals are decomposed
            # ----------------------------------------------------------------
            epn_progress.info("📊 Step 1/3: Checking decompositions...")
            
            epn_required_signals = ['Consumption', 'PV', 'Wind']
            epn_all_decomps = st.session_state.get('all_decompositions', {})
            
            epn_missing = []
            for sig in epn_required_signals:
                key = f"{sig}_{epn_year}"
                if key not in epn_all_decomps:
                    epn_missing.append(sig)
            
            if epn_missing:
                st.warning(f"Missing decompositions: {', '.join(epn_missing)}. Running now...")
                
                # Get reference translations from Consumption if available
                consumption_key = f"Consumption_{epn_year}"
                epn_reference_trans = None
                
                if consumption_key in epn_all_decomps:
                    epn_reference_trans = epn_all_decomps[consumption_key].get('trans')
                
                # Decompose missing signals
                for sig in epn_missing:
                    epn_progress.info(f"📊 Decomposing {sig}...")
                    
                    # Extract signal data for this year
                    years_list = st.session_state['years']
                    year_idx = years_list.index(epn_year)
                    pts_per_year = st.session_state['signal_length']
                    start_idx = year_idx * pts_per_year
                    end_idx = (year_idx + 1) * pts_per_year
                    
                    epn_TS = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                    
                    # Use external translations if available
                    ext_trans = None
                    ref_sig = None
                    if sig != 'Consumption' and epn_reference_trans is not None:
                        ext_trans = epn_reference_trans
                        ref_sig = 'Consumption'
                    
                    # Run decomposition
                    t_file, m_files, r_betas, t_trans = wavelet_decomposition_single_TS(
                        epn_TS,
                        year=epn_year,
                        multi_year=None,
                        country_name=st.session_state['country_name'],
                        signal_type=sig,
                        wl_shape=st.session_state['wavelet_shape'],
                        recompute_translation=False,
                        dpd=st.session_state['dpd'],
                        ndpd=st.session_state['ndpd'],
                        vy=st.session_state['vy'],
                        vw=st.session_state['vw'],
                        vd=st.session_state['vd'],
                        external_translations=ext_trans,
                        reference_signal_type=ref_sig
                    )
                    
                    # Store results
                    decomp_key = f"{sig}_{epn_year}"
                    if 'all_decompositions' not in st.session_state:
                        st.session_state['all_decompositions'] = {}
                    
                    st.session_state['all_decompositions'][decomp_key] = {
                        'results_betas': r_betas,
                        'trans_file': t_file,
                        'matrix_files': m_files,
                        'trans': t_trans
                    }
                    
                    # Update reference for next signals
                    if sig == 'Consumption':
                        epn_reference_trans = t_trans
                
                # Refresh decompositions dict
                epn_all_decomps = st.session_state['all_decompositions']
            
            # ----------------------------------------------------------------
            # STEP B: Get betas and create PMC for scenarios
            # ----------------------------------------------------------------
            epn_progress.info("📊 Step 2/3: Computing PMC for scenarios...")
            
            epn_betas_Load = epn_all_decomps[f'Consumption_{epn_year}']['results_betas']
            epn_betas_PV = epn_all_decomps[f'PV_{epn_year}']['results_betas']
            epn_betas_Wind = epn_all_decomps[f'Wind_{epn_year}']['results_betas']
            
            epn_pmc_list = []
            epn_scenario_names = []
            
            for scen in st.session_state['epn_scenarios_config']:
                pv_share = scen['pv_share']
                scen_name = scen['name']
                
                pmc = [
                    pv_share * np.array(epn_betas_PV[epn_year][i]) + 
                    (1 - pv_share) * np.array(epn_betas_Wind[epn_year][i]) - 
                    np.array(epn_betas_Load[epn_year][i]) 
                    for i in range(len(epn_time_scales))
                ]
                
                epn_pmc_list.append(pmc)
                epn_scenario_names.append(scen_name)
            
            # ----------------------------------------------------------------
            # STEP C: Compute EPN
            # ----------------------------------------------------------------
            epn_progress.info("📊 Step 3/3: Computing EPN metrics...")
            
            epn_Emax, epn_UF, epn_Serv, epn_Pmax = [], [], [], []
            
            for pmc in epn_pmc_list:
                result = calc_epn(pmc, epn_satisfactions, epn_time_scales, epn_dpy, epn_load_factor, shape='square')
                epn_Emax.append(result['emax'])
                epn_UF.append(result['uf'])
                epn_Serv.append(result['serv'])
                epn_Pmax.append(result['pmax'])
            
            # Store results
            st.session_state['epn_computed'] = True
            st.session_state['epn_Emax'] = epn_Emax
            st.session_state['epn_UF'] = epn_UF
            st.session_state['epn_Serv'] = epn_Serv
            st.session_state['epn_Pmax'] = epn_Pmax
            st.session_state['epn_scenario_names'] = epn_scenario_names
            st.session_state['epn_satisfactions'] = epn_satisfactions
            st.session_state['epn_time_scales_result'] = epn_time_scales
            st.session_state['epn_year_result'] = epn_year
            
            epn_progress.empty()
            st.success("✅ EPN Analysis Complete!")
    
    # ========================================================================
    # 6.4 DISPLAY RESULTS
    # ========================================================================
    
    if st.session_state.get('epn_computed', False):
        
        st.markdown("### 📊 EPN Results")
        
        # Retrieve stored results
        disp_Emax = st.session_state['epn_Emax']
        disp_UF = st.session_state['epn_UF']
        disp_Serv = st.session_state['epn_Serv']
        disp_names = st.session_state['epn_scenario_names']
        disp_ts = st.session_state['epn_time_scales_result']
        disp_sat = st.session_state['epn_satisfactions']
        disp_year = st.session_state['epn_year_result']
        
        # Build metrics list
        disp_metrics = []
        if epn_show_energy:
            disp_metrics.append('energy')
        if epn_show_uf:
            disp_metrics.append('uf')
        if epn_show_service:
            disp_metrics.append('service')
        
        if not disp_metrics:
            st.warning("Please select at least one metric to display.")
        else:
            # Plot configuration
            plot_colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377']
            plot_markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross']
            plot_tickvals = [0.75, 3, 10, 24, 168, 720, 8760]
            plot_ticktext = ['0.75', '3', '10', 'day', 'week', 'month', 'year']
            plot_reflines = [24, 168, 720, 8760]
            sat_idx = 0
            
            # ENERGY PLOT
            if 'energy' in disp_metrics:
                fig_energy = go.Figure()
                
                for i, name in enumerate(disp_names):
                    emax_data = disp_Emax[i][:, sat_idx] if disp_Emax[i].ndim > 1 else disp_Emax[i]
                    fig_energy.add_trace(go.Scatter(
                        x=disp_ts, y=emax_data,
                        mode='lines+markers', name=name,
                        line=dict(color=plot_colors[i % len(plot_colors)], width=2),
                        marker=dict(symbol=plot_markers[i % len(plot_markers)], size=10)
                    ))
                
                for xval in plot_reflines:
                    fig_energy.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig_energy.update_layout(
                    title=f"Energy Storage Capacity - {disp_year} ({disp_sat[sat_idx]}% satisfaction)",
                    xaxis_title="Cycle length (h)",
                    yaxis_title="Energy (MWh)",
                    xaxis_type="log",
                    yaxis_type="log",
                    xaxis=dict(tickvals=plot_tickvals, ticktext=plot_ticktext),
                    legend=dict(x=0.02, y=0.98),
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_energy, use_container_width=True, key="epn_energy_chart")
            
            # UF PLOT
            if 'uf' in disp_metrics:
                fig_uf = go.Figure()
                
                for i, name in enumerate(disp_names):
                    uf_data = disp_UF[i][:, sat_idx] if disp_UF[i].ndim > 1 else disp_UF[i]
                    fig_uf.add_trace(go.Scatter(
                        x=disp_ts, y=uf_data,
                        mode='lines+markers', name=name,
                        line=dict(color=plot_colors[i % len(plot_colors)], width=2),
                        marker=dict(symbol=plot_markers[i % len(plot_markers)], size=10)
                    ))
                
                for xval in plot_reflines:
                    fig_uf.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig_uf.update_layout(
                    title=f"Utilization Factor - {disp_year} ({disp_sat[sat_idx]}% satisfaction)",
                    xaxis_title="Cycle length (h)",
                    yaxis_title="Utilization Factor (%)",
                    xaxis_type="log",
                    xaxis=dict(tickvals=plot_tickvals, ticktext=plot_ticktext),
                    yaxis=dict(range=[0, 105]),
                    legend=dict(x=0.02, y=0.98),
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_uf, use_container_width=True, key="epn_uf_chart")
            
            # SERVICE PLOT
            if 'service' in disp_metrics:
                fig_serv = go.Figure()
                
                for i, name in enumerate(disp_names):
                    serv_data = disp_Serv[i][:, sat_idx] if disp_Serv[i].ndim > 1 else disp_Serv[i]
                    fig_serv.add_trace(go.Scatter(
                        x=disp_ts, y=serv_data,
                        mode='lines+markers', name=name,
                        line=dict(color=plot_colors[i % len(plot_colors)], width=2),
                        marker=dict(symbol=plot_markers[i % len(plot_markers)], size=10)
                    ))
                
                for xval in plot_reflines:
                    fig_serv.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig_serv.update_layout(
                    title=f"Service (E × n_cycles) - {disp_year} ({disp_sat[sat_idx]}% satisfaction)",
                    xaxis_title="Cycle length (h)",
                    yaxis_title="E × n_cycles (MWh/year)",
                    xaxis_type="log",
                    xaxis=dict(tickvals=plot_tickvals, ticktext=plot_ticktext),
                    legend=dict(x=0.02, y=0.98),
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_serv, use_container_width=True, key="epn_service_chart")
        
        # Data table
        with st.expander("📋 View EPN Data Table", expanded=False):
            table_rows = []
            for i, name in enumerate(disp_names):
                for j, ts in enumerate(disp_ts):
                    table_rows.append({
                        'Scenario': name,
                        'Time Scale (h)': ts,
                        'Energy (MWh)': disp_Emax[i][j, sat_idx] if disp_Emax[i].ndim > 1 else disp_Emax[i][j],
                        'Power (MW)': st.session_state['epn_Pmax'][i][j, sat_idx] if st.session_state['epn_Pmax'][i].ndim > 1 else st.session_state['epn_Pmax'][i][j],
                        'UF (%)': disp_UF[i][j, sat_idx] if disp_UF[i].ndim > 1 else disp_UF[i][j],
                        'Service (MWh/yr)': disp_Serv[i][j, sat_idx] if disp_Serv[i].ndim > 1 else disp_Serv[i][j]
                    })
            
            epn_df = pd.DataFrame(table_rows)
            st.dataframe(epn_df, use_container_width=True)

# ============================================================================
# END OF STEP 6 - EPN ANALYSIS
# ============================================================================                            

# ============================================================================
# EXPORT WORKFLOW - FIXED VERSION WITH FIGURES
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">📥 Export Workflow</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Export your complete analysis workflow including parameters, results, and all generated visualizations.
    """)
    
    # Default filename
    default_filename = f"wavelet_analysis_{st.session_state.get('country_name', 'region')}_{st.session_state.get('signal_type', 'signal')}_{st.session_state.get('year_to_process', 'year')}"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        export_filename = st.text_input(
            "Export filename (without extension)",
            value=default_filename,
            help="Modify the filename if needed. Extension will be added automatically."
        )
    
    col_html, col_pdf, col_spacer = st.columns([1, 1, 2])
    
    with col_html:
        if st.button("📄 Export as HTML", use_container_width=True):
            with st.spinner("Generating HTML export with figures..."):
                try:
                    import base64
                    from io import BytesIO
                    
                    # Start HTML
                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{export_filename}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #A23B72;
            border-bottom: 2px solid #A23B72;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .info-box {{
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
            margin: 15px 0;
        }}
        .figure {{
            margin: 30px 0;
            page-break-inside: avoid;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
            font-size: 0.9rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>📊 Wavelet Decomposition Analysis Report</h1>
    
    <div class="info-box">
        <strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>Region:</strong> {st.session_state.get('country_name', 'N/A')}<br>
        <strong>Signal:</strong> {st.session_state.get('signal_type', 'N/A')}<br>
        <strong>Year:</strong> {st.session_state.get('year_to_process', 'N/A')}
    </div>
    
    <h2>1. Configuration Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Signal Type</td><td>{st.session_state.get('signal_type', 'N/A')}</td></tr>
        <tr><td>Year</td><td>{st.session_state.get('year_to_process', 'N/A')}</td></tr>
        <tr><td>Region</td><td>{st.session_state.get('country_name', 'N/A')}</td></tr>
        <tr><td>Wavelet Shape</td><td>{st.session_state.get('wavelet_shape', 'N/A')}</td></tr>
        <tr><td>Data Points/Day</td><td>{st.session_state.get('ndpd', 'N/A')}</td></tr>
        <tr><td>Days/Year</td><td>{st.session_state.get('dpy', 'N/A')}</td></tr>
        <tr><td>Time Scales</td><td>15 (0.75h to 8760h)</td></tr>
    </table>
    
    <h2>2. Time Scales</h2>
    <div class="info-box">
        <strong>15 Time Scales:</strong><br>
        {', '.join([f'{ts}h' for ts in st.session_state.get('time_scales', [])])}
    </div>
"""
                    
                    # Add Step 4 visualizations if any
                    if 'generated_plots' in st.session_state and st.session_state['generated_plots']:
                        html_content += "<h2>3. Step 4: Visualizations</h2>\n"
                        
                        for idx, plot_data in enumerate(st.session_state['generated_plots']):
                            if plot_data['fig'] is not None:
                                # Convert matplotlib figure to base64
                                buf = BytesIO()
                                plot_data['fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                img_base64 = base64.b64encode(buf.read()).decode()
                                buf.close()
                                
                                html_content += f"""
    <div class="figure">
        <img src="data:image/png;base64,{img_base64}" alt="Visualization {idx+1}">
        <div class="figure-caption">Figure {idx+1}: {plot_data.get('title', f'Visualization {idx+1}')}</div>
    </div>
"""
                    
                    # Add Step 5 reconstructions if any
                    if 'generated_reconstructions' in st.session_state and st.session_state['generated_reconstructions']:
                        html_content += "<h2>4. Step 5: Reconstructions</h2>\n"
                        
                        for idx, recon_data in enumerate(st.session_state['generated_reconstructions']):
                            if recon_data['fig'] is not None:
                                # Convert Plotly figure to static image
                                img_bytes = recon_data['fig'].to_image(format="png", width=1200, height=600)
                                img_base64 = base64.b64encode(img_bytes).decode()
                                
                                html_content += f"""
    <div class="figure">
        <img src="data:image/png;base64,{img_base64}" alt="Reconstruction {idx+1}">
        <div class="figure-caption">Reconstruction {idx+1}: {recon_data.get('title', f'Reconstruction {idx+1}')}</div>
    </div>
"""
                    
                    # Add methodology reference
                    html_content += """
    <h2>5. Methodology Reference</h2>
    <div class="info-box">
        <strong>Based on:</strong> A. Clerjon and F. Perdu, 
        "Matching intermittency and electricity storage characteristics through 
        time scale analysis: an energy return on investment comparison", 
        <em>Energy Environ. Sci.</em>, 2019, 12, 693-705
    </div>
    
    <div class="footer">
        <p>📊 Wavelet Decomposition Analysis Interface</p>
        <p>Generated with Streamlit - Based on Clerjon & Perdu (2019) methodology</p>
    </div>
</body>
</html>
"""
                    
                    # Offer download
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=html_content,
                        file_name=f"{export_filename}.html",
                        mime="text/html"
                    )
                    
                    st.success("✅ HTML export ready with all figures!")
                    
                    # Show summary
                    n_viz = len(st.session_state.get('generated_plots', []))
                    n_recon = len(st.session_state.get('generated_reconstructions', []))
                    st.info(f"Exported: {n_viz} visualizations + {n_recon} reconstructions")
                    
                except Exception as e:
                    st.error(f"❌ Error generating HTML: {str(e)}")
                    st.exception(e)
    
    with col_pdf:
        if st.button("📕 Export as PDF", use_container_width=True):
            with st.spinner("Generating PDF export..."):
                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
                    from reportlab.lib import colors
                    from reportlab.lib.enums import TA_CENTER
                    import base64
                    
                    # Create PDF buffer
                    buffer = BytesIO()
                    
                    # Create document
                    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
                    
                    # Container for elements
                    elements = []
                    
                    # Styles
                    styles = getSampleStyleSheet()
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        textColor=colors.HexColor('#2E86AB'),
                        spaceAfter=30,
                        alignment=TA_CENTER
                    )
                    heading_style = ParagraphStyle(
                        'CustomHeading',
                        parent=styles['Heading2'],
                        fontSize=16,
                        textColor=colors.HexColor('#A23B72'),
                        spaceAfter=12,
                        spaceBefore=12
                    )
                    
                    # Title
                    elements.append(Paragraph("Wavelet Decomposition Analysis Report", title_style))
                    elements.append(Spacer(1, 0.2*inch))
                    
                    # Metadata
                    elements.append(Paragraph(f"<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
                    elements.append(Paragraph(f"<b>Region:</b> {st.session_state.get('country_name', 'N/A')}", styles['Normal']))
                    elements.append(Paragraph(f"<b>Signal:</b> {st.session_state.get('signal_type', 'N/A')}", styles['Normal']))
                    elements.append(Paragraph(f"<b>Year:</b> {st.session_state.get('year_to_process', 'N/A')}", styles['Normal']))
                    elements.append(Spacer(1, 0.3*inch))
                    
                    # Configuration table
                    elements.append(Paragraph("1. Configuration Parameters", heading_style))
                    
                    config_data = [
                        ['Parameter', 'Value'],
                        ['Signal Type', str(st.session_state.get('signal_type', 'N/A'))],
                        ['Year', str(st.session_state.get('year_to_process', 'N/A'))],
                        ['Wavelet Shape', str(st.session_state.get('wavelet_shape', 'N/A'))],
                        ['Sampling Rate', f"{st.session_state.get('ndpd', 'N/A')} points/day"],
                        ['Time Scales', '15 (0.75h to 8760h)'],
                    ]
                    
                    table = Table(config_data, colWidths=[3*inch, 3*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]))
                    
                    elements.append(table)
                    elements.append(Spacer(1, 0.3*inch))
                    
                    # Add visualizations
                    if 'generated_plots' in st.session_state and st.session_state['generated_plots']:
                        elements.append(Paragraph("2. Visualizations", heading_style))
                        
                        for idx, plot_data in enumerate(st.session_state['generated_plots']):
                            if plot_data['fig'] is not None:
                                # Convert to image
                                buf = BytesIO()
                                plot_data['fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                                buf.seek(0)
                                
                                # Add to PDF
                                img = Image(buf, width=6*inch, height=4*inch)
                                elements.append(img)
                                elements.append(Paragraph(f"<i>Figure {idx+1}: {plot_data.get('title', 'Visualization')}</i>", styles['Normal']))
                                elements.append(Spacer(1, 0.2*inch))
                    
                    # Add reconstructions
                    if 'generated_reconstructions' in st.session_state and st.session_state['generated_reconstructions']:
                        elements.append(Paragraph("3. Reconstructions", heading_style))
                        
                        for idx, recon_data in enumerate(st.session_state['generated_reconstructions']):
                            if recon_data['fig'] is not None:
                                # Convert Plotly to image
                                img_bytes = recon_data['fig'].to_image(format="png", width=1200, height=600)
                                buf = BytesIO(img_bytes)
                                
                                # Add to PDF
                                img = Image(buf, width=6*inch, height=3*inch)
                                elements.append(img)
                                elements.append(Paragraph(f"<i>Reconstruction {idx+1}</i>", styles['Normal']))
                                elements.append(Spacer(1, 0.2*inch))
                    
                    # Methodology
                    elements.append(Paragraph("4. Methodology Reference", heading_style))
                    elements.append(Paragraph(
                        '<b>Based on:</b> A. Clerjon and F. Perdu, "Matching intermittency and '
                        'electricity storage characteristics through time scale analysis: an energy '
                        'return on investment comparison", <i>Energy Environ. Sci.</i>, 2019, 12, 693-705',
                        styles['Normal']
                    ))
                    
                    # Build PDF
                    doc.build(elements)
                    
                    # Get PDF data
                    pdf_data = buffer.getvalue()
                    buffer.close()
                    
                    # Offer download
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_data,
                        file_name=f"{export_filename}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success("✅ PDF export ready with all figures!")
                    
                except ImportError:
                    st.error("❌ reportlab library not installed")
                    st.info("Install with: pip install reportlab kaleido")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
📊 Wavelet Decomposition Analysis Interface | Based on Clerjon & Perdu (2019) methodology
</div>
""", unsafe_allow_html=True)