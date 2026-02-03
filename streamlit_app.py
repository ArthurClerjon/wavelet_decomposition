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
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
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
2. 🎯 Select signal type and year to analyze
3. ⚙️ Configure decomposition parameters
4. 📈 Visualize results (heatmap, FFT spectrum)
5. 🔄 Reconstruct signal with selected time scales
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## 🎛️ Configuration")

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
# STEP 2: SIGNAL SELECTION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🎯 Step 2: Select Signal and Year</div>', unsafe_allow_html=True)
    
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
    
    # ============================================================================
    # STEP 3: DECOMPOSITION PARAMETERS
    # ============================================================================
    
    st.markdown('<div class="section-header">⚙️ Step 3: Decomposition Parameters</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Configure the wavelet decomposition levels. Higher values provide finer resolution 
    but increase computation time.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vy = st.slider("Yearly wavelets", 1, 10, 6, help="Number of yearly wavelet levels")
    with col2:
        vw = st.slider("Weekly wavelets", 1, 10, 3, help="Number of weekly wavelet levels")
    with col3:
        vd = st.slider("Daily wavelets", 1, 10, 6, help="Number of daily wavelet levels")
    
    # Time scales
    time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 
                   273.75, 547.5, 1095., 2190., 4380., 8760.]
    
    st.markdown(f"""
    <div class="info-box">
    <b>Time scales:</b> {len(time_scales)} scales from {time_scales[0]}h to {time_scales[-1]}h (1 year)
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================================
    # STEP 4: RUN DECOMPOSITION
    # ============================================================================
    
    st.markdown('<div class="section-header">🚀 Step 4: Run Wavelet Decomposition</div>', unsafe_allow_html=True)
    
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
                    wl_shape='square',
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
                st.session_state['signal_type'] = signal_type
                st.session_state['year_to_process'] = year_to_process
                st.session_state['country_name'] = country_name
                st.session_state['vy'] = vy
                st.session_state['vw'] = vw
                st.session_state['vd'] = vd
                st.session_state['time_scales'] = time_scales
                
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Decomposition complete!</b><br>
                - Translation file: {trans_file}<br>
                - Matrix file: {matrix_files[0]}<br>
                - Betas computed for {year_to_process}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during decomposition: {str(e)}")
                st.exception(e)

# ============================================================================
# STEP 5: VISUALIZATION OPTIONS
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">📈 Step 5: Visualization</div>', unsafe_allow_html=True)
    
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
        
        # Select which time scales to display
        st.markdown("#### Time Scales to Display")
        
        display_options = st.multiselect(
            "Select time scales (leave empty for all)",
            options=[f"{ts}h" for ts in st.session_state['time_scales']],
            default=["24.0h", "168.0h", "8760.0h"],
            help="Choose which time scales to show in the heatmap"
        )
        
        if display_options:
            reconstructed_time_scales = [float(opt.replace('h', '')) for opt in display_options]
        else:
            reconstructed_time_scales = st.session_state['time_scales']
    
    # Generate visualizations
    if st.button("📊 Generate Visualizations"):
        
        # Plot heatmap
        if plot_heatmap:
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
                        reconstructed_time_scales=reconstructed_time_scales,
                        cmin=cmin,
                        cmax=cmax,
                        ccenter=ccenter
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
# STEP 6: SIGNAL RECONSTRUCTION
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">🔄 Step 6: Signal Reconstruction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Reconstruct the signal using selected time scales. This allows you to filter 
    and analyze specific frequency components.
    """)
    
    # Time scale selection for reconstruction
    st.markdown("#### Select Time Scales for Reconstruction")
    
    # Create checkboxes for each time scale
    time_scale_labels = {
        0.75: "0.75h (45 min)",
        1.5: "1.5h (90 min)",
        3.0: "3h",
        6.0: "6h",
        12.0: "12h (half-day)",
        24.0: "24h (day)",
        42.0: "42h",
        84.0: "84h (3.5 days)",
        168.0: "168h (week)",
        273.75: "273.75h",
        547.5: "547.5h",
        1095.0: "1095h (~45 days)",
        2190.0: "2190h (~91 days)",
        4380.0: "4380h (~6 months)",
        8760.0: "8760h (year)"
    }
    
    # Group time scales by category
    st.markdown("**Sub-daily scales:**")
    subdaily_cols = st.columns(6)
    subdaily_scales = [0.75, 1.5, 3.0, 6.0, 12.0, 24.0]
    subdaily_selected = []
    for i, ts in enumerate(subdaily_scales):
        with subdaily_cols[i]:
            if st.checkbox(time_scale_labels[ts], key=f"ts_{ts}"):
                subdaily_selected.append(ts)
    
    st.markdown("**Weekly to monthly scales:**")
    weekly_cols = st.columns(4)
    weekly_scales = [42.0, 84.0, 168.0, 273.75, 547.5, 1095.0]
    weekly_selected = []
    for i, ts in enumerate(weekly_scales[:4]):
        with weekly_cols[i]:
            if st.checkbox(time_scale_labels[ts], key=f"ts_{ts}"):
                weekly_selected.append(ts)
    
    st.markdown("**Seasonal scales:**")
    seasonal_cols = st.columns(3)
    seasonal_scales = [2190.0, 4380.0, 8760.0]
    seasonal_selected = []
    for i, ts in enumerate(seasonal_scales):
        with seasonal_cols[i]:
            if st.checkbox(time_scale_labels[ts], key=f"ts_{ts}", value=(ts == 8760.0)):
                seasonal_selected.append(ts)
    
    # Combine selected scales
    selected_time_scales = subdaily_selected + weekly_selected + seasonal_selected
    
    if selected_time_scales:
        st.info(f"✅ Selected {len(selected_time_scales)} time scales: {', '.join([f'{ts}h' for ts in selected_time_scales])}")
    else:
        st.warning("⚠️ No time scales selected. Select at least one time scale to reconstruct.")
    
    # Reconstruction options
    add_offset = st.checkbox(
        "Add offset (DC component)",
        value=False,
        help="Include the mean value in reconstruction"
    )
    
    # Run reconstruction
    if st.button("🔄 Reconstruct Signal") and selected_time_scales:
        with st.spinner("Reconstructing signal..."):
            try:
                # Load matrix
                file_mgr = WaveletFileManager(region=st.session_state['country_name'])
                matrix_file = file_mgr.get_matrix_path(st.session_state['year_to_process'])
                matrix = sparse.load_npz(matrix_file)
                
                # Reconstruct
                reconstructed_signal = reconstruct(
                    time_scales=st.session_state['time_scales'],
                    reconstructed_time_scales=selected_time_scales,
                    matrix=matrix,
                    vec_betas=st.session_state['results_betas'][st.session_state['year_to_process']],
                    title=f'{st.session_state["signal_type"]} Signal - Reconstructed with {len(selected_time_scales)} time scales',
                    xmin=0,
                    xmax=st.session_state['dpy'],
                    dpy=st.session_state['dpy'],
                    dpd=st.session_state['ndpd'],
                    add_offset=add_offset,
                    plot=True
                )
                
                # Display reconstructed signal
                st.markdown("### Reconstructed Signal")
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(reconstructed_signal, linewidth=1, color='#2E86AB')
                ax.set_xlabel('Time (data points)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Reconstructed {st.session_state["signal_type"]} Signal')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.success(f"✅ Signal reconstructed with {len(selected_time_scales)} time scales!")
                
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
