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
                trans_file, matrix_files, results_betas = wavelet_decomposition_single_TS(
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
    
    # Store the initial decomposition result
    initial_key = f"{st.session_state['signal_type']}_{st.session_state['year_to_process']}"
    if initial_key not in st.session_state['all_decompositions']:
        st.session_state['all_decompositions'][initial_key] = {
            'results_betas': st.session_state['results_betas'],
            'trans_file': st.session_state['trans_file'],
            'matrix_files': st.session_state['matrix_files']
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
            trans_file, matrix_files, results_betas = wavelet_decomposition_single_TS(
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
                'matrix_files': matrix_files
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
# STEP 5: SIGNAL RECONSTRUCTION WITH FIXES
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">🔄 Step 5: Signal Reconstruction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Reconstruct the signal using selected time scales. This allows you to filter 
    and analyze specific frequency components.
    """)
    
    # Time scale selection for reconstruction
    st.markdown("#### Select Time Scales for Reconstruction")
    
    # Time scale info (shortened labels for single line)
    time_scale_info = {
        0.75: "0.75h", 
        1.5: "1.5h", 
        3.0: "3h", 
        6.0: "6h", 
        12.0: "12h", 
        24.0: "24h",
        42.0: "42h", 
        84.0: "84h", 
        168.0: "168h", 
        273.75: "273.75h", 
        547.5: "547.5h",
        1095.0: "1095h", 
        2190.0: "2190h", 
        4380.0: "4380h", 
        8760.0: "8760h"
    }
    
    # Initialize session state for reconstruction checkboxes if not exists
    if 'reconstruction_checkboxes' not in st.session_state:
        st.session_state['reconstruction_checkboxes'] = {ts: True for ts in st.session_state['time_scales']}
    
    # Select All / Deselect All buttons
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 3])
    
    with col_btn1:
        if st.button("✅ Select All", key="select_all_recon"):
            for ts in st.session_state['time_scales']:
                st.session_state['reconstruction_checkboxes'][ts] = True
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Deselect All", key="deselect_all_recon"):
            for ts in st.session_state['time_scales']:
                st.session_state['reconstruction_checkboxes'][ts] = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("**All Time Scales:**")
    
    # All 15 checkboxes on a single line - FIXED VERSION
    checkbox_cols = st.columns(15)
    
    for i, ts in enumerate(st.session_state['time_scales']):
        with checkbox_cols[i]:
            # Read current value from session state
            current_value = st.session_state['reconstruction_checkboxes'].get(ts, True)
            
            # Create checkbox
            new_value = st.checkbox(
                time_scale_info[ts],
                value=current_value,
                key=f"ts_recon_{ts}",
                label_visibility="visible"
            )
            
            # Only update session state if value changed
            if new_value != current_value:
                st.session_state['reconstruction_checkboxes'][ts] = new_value
    
    # Get selected time scales
    selected_time_scales_recon = [
        ts for ts in st.session_state['time_scales'] 
        if st.session_state['reconstruction_checkboxes'].get(ts, True)
    ]
    
    if selected_time_scales_recon:
        st.info(f"✅ Selected {len(selected_time_scales_recon)} time scales for reconstruction")
    else:
        st.warning("⚠️ No time scales selected. Select at least one time scale to reconstruct.")
    
    # Reconstruction options
    add_offset = st.checkbox(
        "Add offset (DC component)",
        value=False,
        help="Include the mean value in reconstruction"
    )
    
    # Run reconstruction - WITH FIXES FOR INFINITE LOOP
    if st.button("🔄 Reconstruct Signal") and selected_time_scales_recon:
        with st.spinner("Reconstructing signal (this may take a moment)..."):
            try:
                # Load matrix
                file_mgr = WaveletFileManager(
                    region=st.session_state['country_name'],
                    wl_shape=st.session_state['wavelet_shape']
                )
                matrix_file = file_mgr.get_matrix_path(st.session_state['year_to_process'])
                matrix = sparse.load_npz(matrix_file)
                
                # CRITICAL FIX: Parameter name is beta_sheet, NOT vec_betas
                # Also add validation to prevent infinite loop
                reconstructed_signal = reconstruct(
                    time_scales=st.session_state['time_scales'],
                    reconstructed_time_scales=selected_time_scales_recon,
                    matrix=matrix,
                    beta_sheet=st.session_state['results_betas'][st.session_state['year_to_process']],
                    title=f'{st.session_state["signal_type"]} Signal - Reconstructed',
                    xmin=0,
                    xmax=st.session_state['dpy'],
                    dpy=st.session_state['dpy'],
                    dpd=st.session_state['ndpd'],
                    add_offset=add_offset,
                    plot=False  # Don't plot internally, we'll use Plotly
                )
                
                # Validate result to catch infinite loop issues
                if reconstructed_signal is None:
                    st.error("❌ Reconstruction returned None. Check the reconstruct() function.")
                    st.stop()
                
                if len(reconstructed_signal) == 0:
                    st.error("❌ Reconstruction returned empty array.")
                    st.stop()
                
                # Show success message
                st.write(f"✅ Reconstruction complete. Signal length: {len(reconstructed_signal)} points")
                
                # Display reconstructed signal with Plotly
                st.markdown("### Reconstructed Signal")
                
                time_axis = np.linspace(0, st.session_state['dpy'], len(reconstructed_signal))
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=reconstructed_signal,
                        mode='lines',
                        name='Reconstructed Signal',
                        line=dict(color='#2E86AB', width=1.5)
                    )
                )
                
                fig.update_layout(
                    title=f'Reconstructed {st.session_state["signal_type"]} Signal - {st.session_state["country_name"]} ({len(selected_time_scales_recon)} time scales)',
                    xaxis_title='Time (days)',
                    yaxis_title='Amplitude',
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Signal reconstructed with {len(selected_time_scales_recon)} time scales!")
                
                # Show statistics
                st.markdown("#### Reconstruction Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{np.mean(reconstructed_signal):.4f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(reconstructed_signal):.4f}")
                with col3:
                    st.metric("Min", f"{np.min(reconstructed_signal):.4f}")
                with col4:
                    st.metric("Max", f"{np.max(reconstructed_signal):.4f}")
                
            except Exception as e:
                st.error(f"❌ Error during reconstruction: {str(e)}")
                st.exception(e)
                st.stop()
    
    # ========================================================================
    # ADD RECONSTRUCTION SUBPLOTS
    # ========================================================================
    
    if 'reconstruction_subplots' not in st.session_state:
        st.session_state['reconstruction_subplots'] = []
    
    st.markdown("---")
    st.markdown("#### Add More Reconstructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Add Reconstruction Below", key="add_recon_below"):
            st.session_state['reconstruction_subplots'].append({
                'id': len(st.session_state['reconstruction_subplots']),
                'time_scales': list(st.session_state['time_scales']),
                'add_offset': False,
                'position': 'below'
            })
            st.info("Configure the new reconstruction below")
    
    with col2:
        if st.button("➕ Add Reconstruction Right", key="add_recon_right"):
            st.session_state['reconstruction_subplots'].append({
                'id': len(st.session_state['reconstruction_subplots']),
                'time_scales': list(st.session_state['time_scales']),
                'add_offset': False,
                'position': 'right'
            })
            st.info("Configure the new reconstruction below")
    
    # Display and configure additional reconstructions
    if st.session_state['reconstruction_subplots']:
        st.markdown("##### Additional Reconstructions")
        
        for idx, recon_cfg in enumerate(st.session_state['reconstruction_subplots']):
            with st.expander(f"Reconstruction {idx + 2} (Position: {recon_cfg['position']})", expanded=False):
                
                # Time scale selection
                st.markdown("**Select Time Scales:**")
                
                # All checkboxes on a single line
                checkbox_cols_extra = st.columns(15)
                
                for i, ts in enumerate(st.session_state['time_scales']):
                    with checkbox_cols_extra[i]:
                        is_selected = st.checkbox(
                            f"{ts}h",
                            value=(ts in recon_cfg['time_scales']),
                            key=f"ts_extra_{idx}_{ts}"
                        )
                        
                        # Update the config
                        if is_selected and ts not in recon_cfg['time_scales']:
                            recon_cfg['time_scales'].append(ts)
                        elif not is_selected and ts in recon_cfg['time_scales']:
                            recon_cfg['time_scales'].remove(ts)
                
                # Add offset option
                recon_cfg['add_offset'] = st.checkbox(
                    "Add offset",
                    value=recon_cfg['add_offset'],
                    key=f"offset_extra_{idx}"
                )
                
                # Reconstruct button
                if st.button(f"🔄 Reconstruct {idx + 2}", key=f"recon_btn_{idx}"):
                    if recon_cfg['time_scales']:
                        try:
                            # Load matrix
                            file_mgr = WaveletFileManager(
                                region=st.session_state['country_name'],
                                wl_shape=st.session_state['wavelet_shape']
                            )
                            matrix_file = file_mgr.get_matrix_path(st.session_state['year_to_process'])
                            matrix = sparse.load_npz(matrix_file)
                            
                            # Reconstruct
                            reconstructed_signal_extra = reconstruct(
                                time_scales=st.session_state['time_scales'],
                                reconstructed_time_scales=recon_cfg['time_scales'],
                                matrix=matrix,
                                beta_sheet=st.session_state['results_betas'][st.session_state['year_to_process']],
                                title=f'Reconstruction {idx + 2}',
                                xmin=0,
                                xmax=st.session_state['dpy'],
                                dpy=st.session_state['dpy'],
                                dpd=st.session_state['ndpd'],
                                add_offset=recon_cfg['add_offset'],
                                plot=False
                            )
                            
                            # Plot with Plotly
                            time_axis = np.linspace(0, st.session_state['dpy'], len(reconstructed_signal_extra))
                            
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis,
                                    y=reconstructed_signal_extra,
                                    mode='lines',
                                    name=f'Reconstruction {idx + 2}',
                                    line=dict(color='#A23B72', width=1.5)
                                )
                            )
                            
                            fig.update_layout(
                                title=f'Reconstruction {idx + 2} - {len(recon_cfg["time_scales"])} scales',
                                xaxis_title='Time (days)',
                                yaxis_title='Amplitude',
                                height=400,
                                template='plotly_white',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.success(f"✅ Reconstruction {idx + 2} complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.exception(e)
                    else:
                        st.warning("⚠️ Select at least one time scale")
                
                # Remove button
                if st.button(f"🗑️ Remove Reconstruction {idx + 2}", key=f"remove_recon_{idx}"):
                    st.session_state['reconstruction_subplots'].pop(idx)
                    st.rerun()

# ============================================================================
# EXPORT WORKFLOW
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">📥 Export Workflow</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Export your complete analysis workflow including parameters, results, and visualizations.
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
            with st.spinner("Generating HTML export..."):
                try:
                    # Generate HTML content
                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{export_filename}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
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
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .param-table th, .param-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .param-table th {{
            background-color: #2E86AB;
            color: white;
        }}
        .param-table tr:nth-child(even) {{
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
        <strong>Filename:</strong> {export_filename}
    </div>
    
    <h2>1. Data Information</h2>
    <table class="param-table">
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Region</td>
            <td>{st.session_state.get('country_name', 'N/A')}</td>
        </tr>
        <tr>
            <td>Signal Type</td>
            <td>{st.session_state.get('signal_type', 'N/A')}</td>
        </tr>
        <tr>
            <td>Year Analyzed</td>
            <td>{st.session_state.get('year_to_process', 'N/A')}</td>
        </tr>
        <tr>
            <td>Available Years</td>
            <td>{', '.join(st.session_state.get('years', []))}</td>
        </tr>
        <tr>
            <td>Data Points per Day (original)</td>
            <td>{st.session_state.get('dpd', 'N/A')}</td>
        </tr>
        <tr>
            <td>Data Points per Day (interpolated)</td>
            <td>{st.session_state.get('ndpd', 'N/A')}</td>
        </tr>
        <tr>
            <td>Days per Year</td>
            <td>{st.session_state.get('dpy', 'N/A')}</td>
        </tr>
        <tr>
            <td>Total Signal Length</td>
            <td>{st.session_state.get('signal_length', 'N/A'):,} points</td>
        </tr>
    </table>
    
    <h2>2. Decomposition Configuration</h2>
    <table class="param-table">
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Wavelet Shape</td>
            <td>{st.session_state.get('wavelet_shape', 'N/A')}</td>
        </tr>
        <tr>
            <td>Yearly Wavelets (vy)</td>
            <td>{st.session_state.get('vy', 'N/A')}</td>
        </tr>
        <tr>
            <td>Weekly Wavelets (vw)</td>
            <td>{st.session_state.get('vw', 'N/A')}</td>
        </tr>
        <tr>
            <td>Daily Wavelets (vd)</td>
            <td>{st.session_state.get('vd', 'N/A')}</td>
        </tr>
        <tr>
            <td>Total Time Scales</td>
            <td>{len(st.session_state.get('time_scales', []))}</td>
        </tr>
    </table>
    
    <h2>3. Time Scales</h2>
    <div class="info-box">
        <strong>15 Time Scales:</strong><br>
        {', '.join([f'{ts}h' for ts in st.session_state.get('time_scales', [])])}
    </div>
    
    <h2>4. Output Files</h2>
    <table class="param-table">
        <tr>
            <th>File Type</th>
            <th>Path</th>
        </tr>
        <tr>
            <td>Translation File</td>
            <td>{st.session_state.get('trans_file', 'N/A')}</td>
        </tr>
        <tr>
            <td>Matrix Files</td>
            <td>{', '.join(st.session_state.get('matrix_files', ['N/A']))}</td>
        </tr>
    </table>
    
    <h2>5. Methodology</h2>
    <p>
        This analysis follows the wavelet decomposition methodology described in:
    </p>
    <div class="info-box">
        <strong>Reference:</strong> A. Clerjon and F. Perdu, 
        "Matching intermittency and electricity storage characteristics through 
        time scale analysis: an energy return on investment comparison", 
        <em>Energy Environ. Sci.</em>, 2019, 12, 693-705
    </div>
    
    <div class="footer">
        <p>📊 Wavelet Decomposition Analysis Interface</p>
        <p>Based on Clerjon & Perdu (2019) methodology</p>
    </div>
</body>
</html>
"""
                    
                    # Offer download
                    st.download_button(
                        label="⬇️ Download HTML",
                        data=html_content,
                        file_name=f"{export_filename}.html",
                        mime="text/html"
                    )
                    
                    st.success("✅ HTML export ready!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating HTML: {str(e)}")
                    st.exception(e)
    
    with col_pdf:
        if st.button("📕 Export as PDF", use_container_width=True):
            st.info("PDF export requires reportlab library. Install with: pip install reportlab")
            st.markdown("""
            For PDF export functionality, you can:
            1. Install reportlab: `pip install reportlab`
            2. Use the HTML export and convert to PDF using your browser's print function
            3. Or use an online HTML to PDF converter
            """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
📊 Wavelet Decomposition Analysis Interface | Based on Clerjon & Perdu (2019) methodology
</div>
""", unsafe_allow_html=True)