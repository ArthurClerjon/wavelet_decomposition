"""
Wavelet Decomposition Analysis Interface
==========================================
Interactive Streamlit app for analyzing time series using wavelet decomposition.

Based on the Clerjon & Perdu (2019) methodology.

COMPLETE VERSION WITH ALL IMPROVEMENTS APPLIED
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
# STEP 2: SIGNAL SELECTION AND VISUALIZATION
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
    # VISUALIZATION WITH PLOTLY - IMPROVED
    # ========================================================================
    
    st.markdown("### 📊 Time Series Visualization")
    
    # Layout and signal selection
    col_layout, col_signals = st.columns([1, 3])
    
    with col_layout:
        plot_layout = st.radio(
            "Plot layout",
            options=["Combined", "Subplots"],
            help="Combined: All signals on same plot. Subplots: Each signal separate."
        )
    
    with col_signals:
        # Select signals to plot
        signals_to_plot = st.multiselect(
            "Select signals to visualize",
            options=['Consumption', 'Wind', 'PV'],
            default=[signal_type],
            help="Choose one or more signals to compare"
        )
    
    # Year selection for plotting
    years_to_plot = st.multiselect(
        "Select years to plot",
        options=st.session_state['years'],
        default=[year_to_process],
        help="Choose one or more years. Each year will be shown as a separate line."
    )
    
    if signals_to_plot and years_to_plot and st.button("📈 Plot Time Series"):
        with st.spinner("Generating plots..."):
            try:
                # Create time axis (in days)
                points_per_year = st.session_state['signal_length']
                time_axis = np.linspace(0, st.session_state['dpy'], points_per_year)
                
                # Color scheme for signals
                signal_colors = {
                    'Consumption': '#2E86AB',
                    'Wind': '#A23B72',
                    'PV': '#F18F01'
                }
                
                # Line styles for years
                line_styles = ['solid', 'dash', 'dot', 'dashdot']
                
                if plot_layout == "Combined":
                    # ====================================================
                    # COMBINED PLOT - All signals and years on same axes
                    # ====================================================
                    fig = go.Figure()
                    
                    for sig in signals_to_plot:
                        for year_idx, year in enumerate(years_to_plot):
                            # Extract data for this year
                            years_available = st.session_state['years']
                            year_index = years_available.index(year)
                            start_idx = year_index * points_per_year
                            end_idx = (year_index + 1) * points_per_year
                            signal_data = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                            
                            # Create legend name
                            legend_name = f"{sig} ({year})" if len(years_to_plot) > 1 else sig
                            
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
                                    )
                                )
                            )
                    
                    fig.update_layout(
                        title=f"Time Series - {country_name} ({', '.join(years_to_plot)})",
                        xaxis_title="Time (days)",
                        yaxis_title="Normalized Power (MW)",
                        height=500,
                        showlegend=True,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # ====================================================
                    # SUBPLOTS - Each signal separate, all years in each
                    # ====================================================
                    n_plots = len(signals_to_plot)
                    
                    fig = make_subplots(
                        rows=1, 
                        cols=n_plots,
                        subplot_titles=[f"{sig} - {country_name}" for sig in signals_to_plot],
                        horizontal_spacing=0.05
                    )
                    
                    for col_idx, sig in enumerate(signals_to_plot, 1):
                        for year_idx, year in enumerate(years_to_plot):
                            # Extract data for this year
                            years_available = st.session_state['years']
                            year_index = years_available.index(year)
                            start_idx = year_index * points_per_year
                            end_idx = (year_index + 1) * points_per_year
                            signal_data = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                            
                            # Create legend name
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
                                    showlegend=(col_idx == 1)  # Only show legend for first subplot
                                ),
                                row=1, 
                                col=col_idx
                            )
                        
                        # Update axes for each subplot
                        fig.update_xaxes(title_text="Time (days)", row=1, col=col_idx)
                        fig.update_yaxes(title_text="Normalized Power (MW)", row=1, col=col_idx)
                    
                    fig.update_layout(
                        height=500,
                        showlegend=True,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error plotting time series: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # STEP 3: RUN DECOMPOSITION WITH DETAILED INFO
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
    - **Reversible**: Perfect reconstruction possible
    
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
    
    #### Why These Specific Parameters?
    
    - **vy=6**: Optimal coverage of seasonal-to-annual scales
    - **vw=3**: Captures weekly and multi-day patterns
    - **vd=6**: Fine resolution for sub-daily dynamics
    - **Total=15**: Balance between resolution and computation time
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
# STEP 4: VISUALIZATION OPTIONS WITH 15 CHECKBOXES
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">📈 Step 4: Visualization</div>', unsafe_allow_html=True)
    
    st.markdown("Select which visualizations to generate:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        plot_heatmap = st.checkbox(
            "📊 Plot Heatmap",
            value=True,
            help="Display wavelet coefficients as a heatmap"
        )
    
    with col2:
        plot_fft_spectrum = st.checkbox(
            "📉 Plot FFT Spectrum",
            value=False,
            help="Display Fourier transform spectrum"
        )
    
    # Heatmap options
    if plot_heatmap:
        st.markdown("#### Heatmap Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cmin = st.number_input("Color scale minimum", value=-0.1, format="%.3f")
        with col2:
            cmax = st.number_input("Color scale maximum", value=0.1, format="%.3f")
        with col3:
            ccenter = st.number_input(
                "Color scale center",
                value=0.0,
                format="%.3f",
                help="Center of diverging colormap (0 for automatic)"
            )
            if ccenter == 0.0:
                ccenter = None
        
        # ====================================================================
        # TIME SCALE SELECTION WITH 15 CHECKBOXES (SINGLE LINE)
        # ====================================================================
        
        st.markdown("#### Select Time Scales to Display")
        
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
        
        # Initialize session state for checkboxes if not exists
        if 'time_scale_checkboxes' not in st.session_state:
            st.session_state['time_scale_checkboxes'] = {ts: True for ts in st.session_state['time_scales']}
        
        # Select All / Deselect All buttons
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 3])
        
        with col_btn1:
            if st.button("✅ Select All", key="select_all_viz"):
                # Update all checkboxes to True
                for ts in st.session_state['time_scales']:
                    st.session_state['time_scale_checkboxes'][ts] = True
                st.rerun()
        
        with col_btn2:
            if st.button("❌ Deselect All", key="deselect_all_viz"):
                # Update all checkboxes to False
                for ts in st.session_state['time_scales']:
                    st.session_state['time_scale_checkboxes'][ts] = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("**All Time Scales:**")
        
        # All 15 checkboxes on a single line
        checkbox_cols = st.columns(15)
        
        for i, ts in enumerate(st.session_state['time_scales']):
            with checkbox_cols[i]:
                st.session_state['time_scale_checkboxes'][ts] = st.checkbox(
                    time_scale_info[ts],
                    value=st.session_state['time_scale_checkboxes'].get(ts, True),
                    key=f"ts_viz_{ts}",
                    label_visibility="visible"
                )
        
        # Get selected time scales
        selected_time_scales_viz = [
            ts for ts in st.session_state['time_scales'] 
            if st.session_state['time_scale_checkboxes'].get(ts, True)
        ]
        
        # Display selection count
        if selected_time_scales_viz:
            st.info(f"✅ Selected {len(selected_time_scales_viz)} time scales for heatmap")
        else:
            st.warning("⚠️ No time scales selected. Please select at least one time scale.")
    
    # Generate visualizations
    if st.button("📊 Generate Visualizations"):
        
        # Plot heatmap
        if plot_heatmap and selected_time_scales_viz:
            with st.spinner("Generating heatmap..."):
                try:
                    st.markdown("### Wavelet Coefficients Heatmap")
                    
                    fig = plot_betas_heatmap(
                        results_betas=st.session_state['results_betas'],
                        country_name=st.session_state['country_name'],
                        signal_type=st.session_state['signal_type'],
                        vy=st.session_state['vy'],
                        vw=st.session_state['vw'],
                        vd=st.session_state['vd'],
                        ndpd=st.session_state['ndpd'],
                        dpy=st.session_state['dpy'],
                        year=st.session_state['year_to_process'],
                        years=[st.session_state['year_to_process']],
                        time_scales=st.session_state['time_scales'],
                        reconstructed_time_scales=selected_time_scales_viz,
                        cmin=cmin,
                        cmax=cmax,
                        ccenter=ccenter,
                        wl_shape=st.session_state['wavelet_shape']
                    )
                    
                    st.pyplot(fig)
                    st.success("✅ Heatmap generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating heatmap: {str(e)}")
                    st.exception(e)
        
        # Plot FFT spectrum
        if plot_fft_spectrum:
            with st.spinner("Generating FFT spectrum..."):
                try:
                    st.markdown("### FFT Spectrum")
                    
                    # Get full time series for FFT
                    years_available = st.session_state['years']
                    year_index = years_available.index(st.session_state['year_to_process'])
                    points_per_year = st.session_state['signal_length']
                    start_idx = year_index * points_per_year
                    end_idx = (year_index + 1) * points_per_year
                    
                    input_data = st.session_state['stacked_input_data'][st.session_state['signal_type']][start_idx:end_idx]
                    
                    fig = fft(
                        ndpd=st.session_state['ndpd'],
                        dpy=st.session_state['dpy'],
                        signal_type=st.session_state['signal_type'],
                        year=st.session_state['year_to_process'],
                        input_data=input_data
                    )
                    
                    st.pyplot(fig)
                    st.success("✅ FFT spectrum generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating FFT spectrum: {str(e)}")
                    st.exception(e)

# ============================================================================
# STEP 5: SIGNAL RECONSTRUCTION
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
    
    # All 15 checkboxes on a single line
    checkbox_cols = st.columns(15)
    
    for i, ts in enumerate(st.session_state['time_scales']):
        with checkbox_cols[i]:
            st.session_state['reconstruction_checkboxes'][ts] = st.checkbox(
                time_scale_info[ts],
                value=st.session_state['reconstruction_checkboxes'].get(ts, True),
                key=f"ts_recon_{ts}",
                label_visibility="visible"
            )
    
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
    
    # Run reconstruction
    if st.button("🔄 Reconstruct Signal") and selected_time_scales_recon:
        with st.spinner("Reconstructing signal..."):
            try:
                # Load matrix
                file_mgr = WaveletFileManager(
                    region=st.session_state['country_name'],
                    wl_shape=st.session_state['wavelet_shape']
                )
                matrix_file = file_mgr.get_matrix_path(st.session_state['year_to_process'])
                matrix = sparse.load_npz(matrix_file)
                
                # CRITICAL FIX: Parameter name is beta_sheet, NOT vec_betas
                reconstructed_signal = reconstruct(
                    time_scales=st.session_state['time_scales'],
                    reconstructed_time_scales=selected_time_scales_recon,
                    matrix=matrix,
                    beta_sheet=st.session_state['results_betas'][st.session_state['year_to_process']],  # FIXED
                    title=f'{st.session_state["signal_type"]} Signal - Reconstructed',
                    xmin=0,
                    xmax=st.session_state['dpy'],
                    dpy=st.session_state['dpy'],
                    dpd=st.session_state['ndpd'],
                    add_offset=add_offset,
                    plot=False  # Don't plot internally, we'll use Plotly
                )
                
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

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
📊 Wavelet Decomposition Analysis Interface | Based on Clerjon & Perdu (2019) methodology
</div>
""", unsafe_allow_html=True)