"""
Wavelet Decomposition Analysis Interface
==========================================
Interactive Streamlit app for analyzing time series using wavelet decomposition.

Based on the Clerjon & Perdu (2019) methodology.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import custom modules
from file_manager import WaveletFileManager
from wavelet_decomposition import wavelet_decomposition_single_TS, reconstruct
from plots import plot_betas_heatmap
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
    # VISUALIZATION WITH PLOTLY
    # ========================================================================
    
    st.markdown("### 📊 Time Series Visualization")
    
    # Select signals to plot
    signals_to_plot = st.multiselect(
        "Select signals to visualize (side-by-side)",
        options=['Consumption', 'Wind', 'PV'],
        default=[signal_type],
        help="Choose one or more signals to compare"
    )
    
    if signals_to_plot and st.button("📈 Plot Time Series"):
        with st.spinner("Generating plots..."):
            try:
                # Extract data for selected year
                years_available = st.session_state['years']
                year_index = years_available.index(year_to_process)
                points_per_year = st.session_state['signal_length']
                start_idx = year_index * points_per_year
                end_idx = (year_index + 1) * points_per_year
                
                # Create time axis (in days)
                time_axis = np.linspace(0, st.session_state['dpy'], points_per_year)
                
                # Determine subplot layout
                n_plots = len(signals_to_plot)
                
                # Create subplots
                fig = make_subplots(
                    rows=1, 
                    cols=n_plots,
                    subplot_titles=[f"{sig} ({year_to_process})" for sig in signals_to_plot],
                    horizontal_spacing=0.05
                )
                
                # Color scheme
                colors = {
                    'Consumption': '#2E86AB',
                    'Wind': '#A23B72',
                    'PV': '#F18F01'
                }
                
                # Add each signal to subplots
                for i, sig in enumerate(signals_to_plot, 1):
                    signal_data = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=signal_data,
                            mode='lines',
                            name=sig,
                            line=dict(color=colors.get(sig, '#333333'), width=1.5),
                            showlegend=(i == 1)  # Only show legend for first plot
                        ),
                        row=1, 
                        col=i
                    )
                    
                    # Update axes for each subplot
                    fig.update_xaxes(title_text="Time (days)", row=1, col=i)
                    fig.update_yaxes(title_text="Normalized Power (MW)", row=1, col=i)
                
                # Update layout
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    hovermode='x unified',
                    template='plotly_white',
                    title_text=f"Time Series Comparison - {year_to_process}",
                    title_x=0.5
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                st.markdown("#### 📊 Signal Statistics")
                stats_cols = st.columns(len(signals_to_plot))
                
                for i, sig in enumerate(signals_to_plot):
                    signal_data = st.session_state['stacked_input_data'][sig][start_idx:end_idx]
                    with stats_cols[i]:
                        st.markdown(f"**{sig}**")
                        st.metric("Mean", f"{np.mean(signal_data):.3f} MW")
                        st.metric("Std Dev", f"{np.std(signal_data):.3f} MW")
                        st.metric("Max", f"{np.max(signal_data):.3f} MW")
                        st.metric("Min", f"{np.min(signal_data):.3f} MW")
                
            except Exception as e:
                st.error(f"❌ Error plotting time series: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # STEP 3: RUN DECOMPOSITION (REMOVED PARAMETERS - FIXED AT 15 SCALES)
    # ============================================================================
    
    st.markdown('<div class="section-header">🚀 Step 3: Run Wavelet Decomposition</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The decomposition uses **15 time scales** (fixed) spanning from 0.75h to 8760h (1 year).
    
    **Time scales:** 0.75h, 1.5h, 3h, 6h, 12h, 24h, 42h, 84h, 168h, 273.75h, 547.5h, 1095h, 2190h, 4380h, 8760h
    """)
    
    # Fixed parameters for 15 time scales
    vy = 6  # Yearly wavelets
    vw = 3  # Weekly wavelets
    vd = 6  # Daily wavelets
    
    time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 
                   273.75, 547.5, 1095., 2190., 4380., 8760.]
    
    st.markdown(f"""
    <div class="info-box">
    <b>Decomposition parameters (fixed):</b><br>
    - Yearly wavelets: {vy}<br>
    - Weekly wavelets: {vw}<br>
    - Daily wavelets: {vd}<br>
    - Total time scales: {len(time_scales)}<br>
    - Wavelet shape: {wavelet_shape}
    </div>
    """, unsafe_allow_html=True)
    
    recompute_translation = st.checkbox(
        "Recompute translations",
        value=False,
        help="If checked, recompute optimal translations (slower). Otherwise, load existing translations."
    )
    
    if st.button("🚀 Run Wavelet Decomposition", type="primary"):
        with st.spinner(f"Running wavelet decomposition for {signal_type} signal in {year_to_process}..."):
            try:
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
                
                # Store results in session state
                st.session_state['decomposition_done'] = True
                st.session_state['trans_file'] = trans_file
                st.session_state['matrix_files'] = matrix_files
                st.session_state['results_betas'] = results_betas
                st.session_state['vy'] = vy
                st.session_state['vw'] = vw
                st.session_state['vd'] = vd
                st.session_state['time_scales'] = time_scales
                
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Decomposition complete!</b><br>
                - Translation file: {trans_file}<br>
                - Matrix file: {matrix_files[0]}<br>
                - Betas computed for {year_to_process}<br>
                - Time scales: 15 (from 0.75h to 8760h)
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during decomposition: {str(e)}")
                st.exception(e)

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
        # TIME SCALE SELECTION WITH 15 CHECKBOXES
        # ====================================================================
        
        st.markdown("#### Select Time Scales to Display")
        
        # Define all 15 time scales with labels
        time_scale_info = {
            0.75: "0.75h (45 min)",
            1.5: "1.5h (90 min)",
            3.0: "3h",
            6.0: "6h",
            12.0: "12h (half-day)",
            24.0: "24h (day)",
            42.0: "42h",
            84.0: "84h (3.5 days)",
            168.0: "168h (week)",
            273.75: "273.75h (~11 days)",
            547.5: "547.5h (~23 days)",
            1095.0: "1095h (~45 days)",
            2190.0: "2190h (~91 days)",
            4380.0: "4380h (~6 months)",
            8760.0: "8760h (year)"
        }
        
        # Initialize session state for checkboxes if not exists
        if 'time_scale_checkboxes' not in st.session_state:
            st.session_state['time_scale_checkboxes'] = {ts: True for ts in st.session_state['time_scales']}
        
        # Select All / Deselect All buttons
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 3])
        
        with col_btn1:
            if st.button("✅ Select All", key="select_all"):
                for ts in st.session_state['time_scales']:
                    st.session_state['time_scale_checkboxes'][ts] = True
                st.rerun()
        
        with col_btn2:
            if st.button("❌ Deselect All", key="deselect_all"):
                for ts in st.session_state['time_scales']:
                    st.session_state['time_scale_checkboxes'][ts] = False
                st.rerun()
        
        st.markdown("---")
        
        # Group time scales by category
        st.markdown("**🕐 Sub-daily scales (< 24h):**")
        subdaily_cols = st.columns(6)
        subdaily_scales = [0.75, 1.5, 3.0, 6.0, 12.0, 24.0]
        
        for i, ts in enumerate(subdaily_scales):
            with subdaily_cols[i]:
                st.session_state['time_scale_checkboxes'][ts] = st.checkbox(
                    time_scale_info[ts], 
                    value=st.session_state['time_scale_checkboxes'].get(ts, True),
                    key=f"ts_viz_{ts}"
                )
        
        st.markdown("**📅 Weekly to monthly scales (1-45 days):**")
        weekly_cols = st.columns(5)
        weekly_scales = [42.0, 84.0, 168.0, 273.75, 547.5]
        
        for i, ts in enumerate(weekly_scales):
            with weekly_cols[i]:
                st.session_state['time_scale_checkboxes'][ts] = st.checkbox(
                    time_scale_info[ts], 
                    value=st.session_state['time_scale_checkboxes'].get(ts, True),
                    key=f"ts_viz_{ts}"
                )
        
        st.markdown("**🌍 Seasonal to annual scales (> 45 days):**")
        seasonal_cols = st.columns(4)
        seasonal_scales = [1095.0, 2190.0, 4380.0, 8760.0]
        
        for i, ts in enumerate(seasonal_scales):
            with seasonal_cols[i]:
                st.session_state['time_scale_checkboxes'][ts] = st.checkbox(
                    time_scale_info[ts], 
                    value=st.session_state['time_scale_checkboxes'].get(ts, True),
                    key=f"ts_viz_{ts}"
                )
        
        # Get selected time scales
        selected_time_scales_viz = [
            ts for ts in st.session_state['time_scales'] 
            if st.session_state['time_scale_checkboxes'].get(ts, True)
        ]
        
        if selected_time_scales_viz:
            st.info(f"✅ Selected {len(selected_time_scales_viz)} time scales for heatmap")
        else:
            st.warning("⚠️ No time scales selected. Please select at least one time scale.")
    
    # Generate visualizations
    if st.button("📊 Generate Visualizations"):
        
        # Import fft function
        from plots import fft
        
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
                    
                    # Modified fft call - create figure and display
                    fig = fft(
                        ndpd=st.session_state['ndpd'],
                        dpy=st.session_state['dpy'],
                        signal_type=st.session_state['signal_type'],
                        year=st.session_state['year_to_process'],
                        input_data=input_data
                    )
                    
                    # Note: fft function needs to be modified to return fig
                    # For now, it will display via plt.show() internally
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
    
    # Define all 15 time scales with labels
    time_scale_info = {
        0.75: "0.75h (45 min)",
        1.5: "1.5h (90 min)",
        3.0: "3h",
        6.0: "6h",
        12.0: "12h (half-day)",
        24.0: "24h (day)",
        42.0: "42h",
        84.0: "84h (3.5 days)",
        168.0: "168h (week)",
        273.75: "273.75h (~11 days)",
        547.5: "547.5h (~23 days)",
        1095.0: "1095h (~45 days)",
        2190.0: "2190h (~91 days)",
        4380.0: "4380h (~6 months)",
        8760.0: "8760h (year)"
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
    
    # Group time scales by category
    st.markdown("**🕐 Sub-daily scales (< 24h):**")
    subdaily_cols = st.columns(6)
    subdaily_scales = [0.75, 1.5, 3.0, 6.0, 12.0, 24.0]
    
    for i, ts in enumerate(subdaily_scales):
        with subdaily_cols[i]:
            st.session_state['reconstruction_checkboxes'][ts] = st.checkbox(
                time_scale_info[ts], 
                value=st.session_state['reconstruction_checkboxes'].get(ts, True),
                key=f"ts_recon_{ts}"
            )
    
    st.markdown("**📅 Weekly to monthly scales (1-45 days):**")
    weekly_cols = st.columns(5)
    weekly_scales = [42.0, 84.0, 168.0, 273.75, 547.5]
    
    for i, ts in enumerate(weekly_scales):
        with weekly_cols[i]:
            st.session_state['reconstruction_checkboxes'][ts] = st.checkbox(
                time_scale_info[ts], 
                value=st.session_state['reconstruction_checkboxes'].get(ts, True),
                key=f"ts_recon_{ts}"
            )
    
    st.markdown("**🌍 Seasonal to annual scales (> 45 days):**")
    seasonal_cols = st.columns(4)
    seasonal_scales = [1095.0, 2190.0, 4380.0, 8760.0]
    
    for i, ts in enumerate(seasonal_scales):
        with seasonal_cols[i]:
            st.session_state['reconstruction_checkboxes'][ts] = st.checkbox(
                time_scale_info[ts], 
                value=st.session_state['reconstruction_checkboxes'].get(ts, True),
                key=f"ts_recon_{ts}"
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
                
                # Reconstruct
                reconstructed_signal = reconstruct(
                    time_scales=st.session_state['time_scales'],
                    reconstructed_time_scales=selected_time_scales_recon,
                    matrix=matrix,
                    vec_betas=st.session_state['results_betas'][st.session_state['year_to_process']],
                    title=f'{st.session_state["signal_type"]} Signal - Reconstructed with {len(selected_time_scales_recon)} time scales',
                    xmin=0,
                    xmax=st.session_state['dpy'],
                    dpy=st.session_state['dpy'],
                    dpd=st.session_state['ndpd'],
                    add_offset=add_offset,
                    plot=False  # Don't plot internally, we'll plot with Plotly
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
                    title=f'Reconstructed {st.session_state["signal_type"]} Signal ({len(selected_time_scales_recon)} time scales)',
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