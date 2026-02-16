"""
Wavelet Decomposition Analysis Interface
==========================================
Interactive Streamlit app for analyzing time series using wavelet decomposition.

Based on the Clerjon & Perdu (2019) methodology.

RESTRUCTURED VERSION:
- Step 1: Upload Data
- Step 2: Signal Selection
- Step 3: Run Decomposition
- Step 4: Analysis & Visualization (heatmap, FFT, reconstruction)
- Step 5: EPN Analysis
- Export: HTML report with interactive plots
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
from plots import plot_betas_heatmap, fft, plot_EPN_scenarios_plotly
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
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
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
3. 🚀 Run wavelet decomposition
4. 📈 Analyze & Visualize (heatmap, FFT, reconstruction)
5. ⚡ EPN Analysis (energy mix scenarios)
6. 📥 Export as HTML report
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
    help="Square wavelets are faster. Sine wavelets provide smoother decomposition."
)

st.sidebar.markdown(f"""
<div class="info-box">
<b>Selected:</b> {wavelet_shape}
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: WAVELET DECOMPOSITION EXPLANATION (moved from Step 3)
# ============================================================================

with st.sidebar.expander("ℹ️ About Wavelet Decomposition", expanded=False):
    st.markdown("""
    ### Methodology
    
    Based on **Clerjon & Perdu (2019)** - *Energy Environ. Sci.*, 12, 693-705.
    
    ### 15 Time Scales
    
    **Daily (6 scales):**
    0.75h, 1.5h, 3h, 6h, 12h, 24h
    
    **Weekly (3 scales):**
    42h, 84h, 168h
    
    **Yearly (6 scales):**
    273.75h, 547.5h, 1095h, 2190h, 4380h, 8760h
    
    ### Mathematical Basis
    
    - **Orthogonal**: Each scale is independent
    - **Additive**: Signal = sum of components
    - **Complete**: Full signal reconstruction
    
    ### Haar Wavelet
    ```
    +1 ┌─────┐
       │     │
     0 ┴─────┴─────
                   │
    -1             └─────┐
    ```
    
    ### Process
    1. Translation optimization
    2. Matrix generation (sparse)
    3. Coefficient calculation (LSQR)
    4. Result caching
    """)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'all_decompositions' not in st.session_state:
    st.session_state['all_decompositions'] = {}

if 'generated_plots' not in st.session_state:
    st.session_state['generated_plots'] = []

if 'generated_reconstructions' not in st.session_state:
    st.session_state['generated_reconstructions'] = []

# ============================================================================
# HELPER FUNCTION: Ensure Signal is Decomposed
# ============================================================================

def ensure_decomposition(signal, year):
    """
    Check if decomposition exists for this signal/year.
    If not, run decomposition automatically.
    Returns: (success, results_betas or error_message)
    """
    key = f"{signal}_{year}"
    
    # Check if already decomposed
    if key in st.session_state.get('all_decompositions', {}):
        return True, st.session_state['all_decompositions'][key]['results_betas']
    
    # Need to decompose
    st.info(f"⏳ Auto-decomposing {signal} for {year}...")
    
    try:
        years_available = st.session_state['years']
        year_index = years_available.index(year)
        points_per_year = st.session_state['signal_length']
        start_idx = year_index * points_per_year
        end_idx = (year_index + 1) * points_per_year
        
        TS_single_year = st.session_state['stacked_input_data'][signal][start_idx:end_idx]
        
        trans_file, matrix_files, results_betas, trans = wavelet_decomposition_single_TS(
            TS_single_year,
            year=year,
            multi_year=None,
            country_name=st.session_state['country_name'],
            signal_type=signal,
            wl_shape=st.session_state['wavelet_shape'],
            recompute_translation=False,
            dpd=st.session_state['dpd'],
            ndpd=st.session_state['ndpd'],
            vy=st.session_state['vy'],
            vw=st.session_state['vw'],
            vd=st.session_state['vd']
        )
        
        st.session_state['all_decompositions'][key] = {
            'results_betas': results_betas,
            'trans_file': trans_file,
            'matrix_files': matrix_files,
            'trans': trans
        }
        
        st.success(f"✅ {signal} ({year}) decomposed!")
        return True, results_betas
        
    except Exception as e:
        return False, str(e)

# ============================================================================
# STEP 1: FILE UPLOAD AND DATA IMPORT
# ============================================================================

st.markdown('<div class="section-header">📁 Step 1: Upload Data File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Excel file with time series data",
    type=['xlsx', 'xls'],
    help="File should contain columns: 'Consumption', 'Wind', 'PV'"
)

if uploaded_file is not None:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    st.markdown("### Import Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dpd = st.number_input("Data points/day (original)", min_value=1, max_value=96, value=48)
    
    with col2:
        ndpd = st.number_input("Data points/day (interpolated)", min_value=1, max_value=128, value=64)
    
    with col3:
        dpy = st.number_input("Days per year", min_value=1, max_value=366, value=365)
    
    if st.button("🔄 Import Data"):
        with st.spinner("Importing..."):
            try:
                time_series_options = ['Consumption', 'Wind', 'PV']
                
                stacked_input_data, years = import_excel(
                    "", temp_file_path, dpd, ndpd, dpy, time_series_options, interp=True
                )
                
                st.session_state['data_imported'] = True
                st.session_state['stacked_input_data'] = stacked_input_data
                st.session_state['years'] = years
                st.session_state['dpd'] = dpd
                st.session_state['ndpd'] = ndpd
                st.session_state['dpy'] = dpy
                st.session_state['signal_length'] = ndpd * dpy
                st.session_state['wavelet_shape'] = wavelet_shape
                
                # Fixed parameters
                st.session_state['vy'] = 6
                st.session_state['vw'] = 3
                st.session_state['vd'] = 6
                st.session_state['time_scales'] = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 
                                                   273.75, 547.5, 1095., 2190., 4380., 8760.]
                
                st.success("✅ Data imported!")
                
                st.markdown(f"""
                **Data Info:** {', '.join(time_series_options)} | 
                Years: {', '.join(years)} | 
                {ndpd * dpy:,} points/year
                """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# STEP 2: SIGNAL SELECTION AND VISUALIZATION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🎯 Step 2: Select Signal</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_type = st.selectbox("Signal type", options=['Consumption', 'Wind', 'PV'])
    
    with col2:
        year_to_process = st.selectbox("Year", options=st.session_state['years'])
    
    with col3:
        country_name = st.text_input("Region name", value="France")
    
    st.session_state['signal_type'] = signal_type
    st.session_state['year_to_process'] = year_to_process
    st.session_state['country_name'] = country_name
    st.session_state['wavelet_shape'] = wavelet_shape
    
    # Quick visualization
    if st.checkbox("Show time series preview", value=False):
        years_available = st.session_state['years']
        year_index = years_available.index(year_to_process)
        points_per_year = st.session_state['signal_length']
        start_idx = year_index * points_per_year
        end_idx = (year_index + 1) * points_per_year
        
        signal_data = st.session_state['stacked_input_data'][signal_type][start_idx:end_idx]
        time_axis = np.linspace(0, st.session_state['dpy'], points_per_year)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=signal_data, mode='lines', name=signal_type))
        fig.update_layout(
            title=f"{signal_type} - {year_to_process} ({country_name})",
            xaxis_title="Time (days)",
            yaxis_title="Normalized Power",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# STEP 3: RUN DECOMPOSITION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🚀 Step 3: Run Wavelet Decomposition</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    **Configuration:** {len(st.session_state['time_scales'])} time scales | 
    Shape: {wavelet_shape} | 
    Resolution: {st.session_state['ndpd']} pts/day
    """)
    
    recompute_translation = st.checkbox("Recompute translations (slower)", value=False)
    
    if st.button("🚀 Run Decomposition", type="primary"):
        with st.spinner(f"Decomposing {signal_type} for {year_to_process}..."):
            try:
                years_available = st.session_state['years']
                year_index = years_available.index(year_to_process)
                points_per_year = st.session_state['signal_length']
                start_idx = year_index * points_per_year
                end_idx = (year_index + 1) * points_per_year
                
                TS_single_year = st.session_state['stacked_input_data'][signal_type][start_idx:end_idx]
                
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
                    vy=st.session_state['vy'],
                    vw=st.session_state['vw'],
                    vd=st.session_state['vd']
                )
                
                st.session_state['decomposition_done'] = True
                st.session_state['trans_file'] = trans_file
                st.session_state['matrix_files'] = matrix_files
                st.session_state['results_betas'] = results_betas
                st.session_state['trans'] = trans
                
                # Store in all_decompositions
                decomp_key = f"{signal_type}_{year_to_process}"
                st.session_state['all_decompositions'][decomp_key] = {
                    'results_betas': results_betas,
                    'trans_file': trans_file,
                    'matrix_files': matrix_files,
                    'trans': trans
                }
                
                st.success(f"✅ Decomposition complete! Files saved.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# ============================================================================
# STEP 4: ANALYSIS & VISUALIZATION (Heatmap, FFT, Reconstruction)
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">📈 Step 4: Analysis & Visualization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Create visualizations for any signal/year. Signals will be auto-decomposed if needed.
    """)
    
    # Show decomposition status
    with st.expander("ℹ️ Decomposition Status", expanded=False):
        if st.session_state['all_decompositions']:
            for key in st.session_state['all_decompositions'].keys():
                sig, yr = key.split('_')
                st.write(f"✅ {sig} - {yr}")
        else:
            st.write("No decompositions yet")
    
    # ========================================================================
    # TABS FOR DIFFERENT VISUALIZATIONS
    # ========================================================================
    
    tab_heatmap, tab_fft, tab_recon = st.tabs(["🔥 Heatmap", "📊 FFT Spectrum", "🔄 Reconstruction"])
    
    # ========================================================================
    # TAB 1: HEATMAP
    # ========================================================================
    
    with tab_heatmap:
        st.markdown("### Wavelet Coefficients Heatmap")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hm_signal = st.selectbox("Signal", ['Consumption', 'Wind', 'PV'], key="hm_signal")
        
        with col2:
            hm_year = st.selectbox("Year", st.session_state['years'], key="hm_year")
        
        # Time scale selection
        st.markdown("**Select Time Scales:**")
        
        if 'hm_scales' not in st.session_state:
            st.session_state['hm_scales'] = {ts: True for ts in st.session_state['time_scales']}
        
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("✅ All", key="hm_all"):
                for ts in st.session_state['time_scales']:
                    st.session_state['hm_scales'][ts] = True
                st.rerun()
        with col_btn2:
            if st.button("❌ None", key="hm_none"):
                for ts in st.session_state['time_scales']:
                    st.session_state['hm_scales'][ts] = False
                st.rerun()
        
        # Checkboxes
        checkbox_cols = st.columns(15)
        time_scale_labels = {
            0.75: "0.75h", 1.5: "1.5h", 3.0: "3h", 6.0: "6h", 12.0: "12h", 24.0: "24h",
            42.0: "42h", 84.0: "84h", 168.0: "168h", 273.75: "274h", 547.5: "548h",
            1095.0: "1095h", 2190.0: "2190h", 4380.0: "4380h", 8760.0: "8760h"
        }
        
        for i, ts in enumerate(st.session_state['time_scales']):
            with checkbox_cols[i]:
                current = st.session_state['hm_scales'].get(ts, True)
                new_val = st.checkbox(time_scale_labels[ts], value=current, key=f"hm_ts_{ts}")
                if new_val != current:
                    st.session_state['hm_scales'][ts] = new_val
        
        selected_scales = [ts for ts in st.session_state['time_scales'] if st.session_state['hm_scales'].get(ts, True)]
        
        if selected_scales:
            st.info(f"✅ {len(selected_scales)} time scales selected")
        else:
            st.warning("⚠️ Select at least one time scale")
        
        if st.button("📊 Generate Heatmap", key="gen_heatmap", type="primary") and selected_scales:
            with st.spinner("Generating heatmap..."):
                try:
                    success, results_or_error = ensure_decomposition(hm_signal, hm_year)
                    
                    if not success:
                        st.error(f"Decomposition failed: {results_or_error}")
                    else:
                        results_betas = results_or_error
                        
                        fig = plot_betas_heatmap(
                            results_betas=results_betas,
                            country_name=st.session_state['country_name'],
                            signal_type=hm_signal,
                            vy=st.session_state['vy'],
                            vw=st.session_state['vw'],
                            vd=st.session_state['vd'],
                            ndpd=st.session_state['ndpd'],
                            dpy=st.session_state['dpy'],
                            year=hm_year,
                            years=[hm_year],
                            time_scales=st.session_state['time_scales'],
                            reconstructed_time_scales=selected_scales,
                            cmin=-0.1,
                            cmax=0.1,
                            ccenter=None,
                            wl_shape=st.session_state['wavelet_shape']
                        )
                        
                        st.pyplot(fig)
                        
                        # Store for export
                        st.session_state['generated_plots'].append({
                            'type': 'heatmap',
                            'signal': hm_signal,
                            'year': hm_year,
                            'fig': fig,
                            'title': f"Heatmap: {hm_signal} - {hm_year}",
                            'scales': selected_scales
                        })
                        
                        st.success("✅ Heatmap generated and added to export list")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # ========================================================================
    # TAB 2: FFT
    # ========================================================================
    
    with tab_fft:
        st.markdown("### FFT Spectrum Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fft_signal = st.selectbox("Signal", ['Consumption', 'Wind', 'PV'], key="fft_signal")
        
        with col2:
            fft_year = st.selectbox("Year", st.session_state['years'], key="fft_year")
        
        if st.button("📊 Generate FFT", key="gen_fft", type="primary"):
            with st.spinner("Computing FFT..."):
                try:
                    years_available = st.session_state['years']
                    year_index = years_available.index(fft_year)
                    points_per_year = st.session_state['signal_length']
                    start_idx = year_index * points_per_year
                    end_idx = (year_index + 1) * points_per_year
                    
                    input_data = st.session_state['stacked_input_data'][fft_signal][start_idx:end_idx]
                    
                    fig = fft(
                        ndpd=st.session_state['ndpd'],
                        dpy=st.session_state['dpy'],
                        signal_type=fft_signal,
                        year=fft_year,
                        input_data=input_data
                    )
                    
                    st.pyplot(fig)
                    
                    # Store for export
                    st.session_state['generated_plots'].append({
                        'type': 'fft',
                        'signal': fft_signal,
                        'year': fft_year,
                        'fig': fig,
                        'title': f"FFT: {fft_signal} - {fft_year}"
                    })
                    
                    st.success("✅ FFT generated and added to export list")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # ========================================================================
    # TAB 3: RECONSTRUCTION
    # ========================================================================
    
    with tab_recon:
        st.markdown("### Signal Reconstruction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            recon_signal = st.selectbox("Signal", ['Consumption', 'Wind', 'PV'], key="recon_signal")
        
        with col2:
            recon_year = st.selectbox("Year", st.session_state['years'], key="recon_year")
        
        # Time scale selection
        st.markdown("**Select Time Scales for Reconstruction:**")
        
        if 'recon_scales' not in st.session_state:
            st.session_state['recon_scales'] = {ts: True for ts in st.session_state['time_scales']}
        
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("✅ All", key="recon_all"):
                for ts in st.session_state['time_scales']:
                    st.session_state['recon_scales'][ts] = True
                st.rerun()
        with col_btn2:
            if st.button("❌ None", key="recon_none"):
                for ts in st.session_state['time_scales']:
                    st.session_state['recon_scales'][ts] = False
                st.rerun()
        
        # Checkboxes
        checkbox_cols = st.columns(15)
        for i, ts in enumerate(st.session_state['time_scales']):
            with checkbox_cols[i]:
                current = st.session_state['recon_scales'].get(ts, True)
                new_val = st.checkbox(time_scale_labels[ts], value=current, key=f"recon_ts_{ts}")
                if new_val != current:
                    st.session_state['recon_scales'][ts] = new_val
        
        selected_recon_scales = [ts for ts in st.session_state['time_scales'] if st.session_state['recon_scales'].get(ts, True)]
        
        if selected_recon_scales:
            st.info(f"✅ {len(selected_recon_scales)} time scales selected")
        else:
            st.warning("⚠️ Select at least one time scale")
        
        add_offset = st.checkbox("Add offset (DC component)", value=False, key="recon_offset")
        
        if st.button("🔄 Generate Reconstruction", key="gen_recon", type="primary") and selected_recon_scales:
            with st.spinner("Reconstructing signal..."):
                try:
                    success, results_or_error = ensure_decomposition(recon_signal, recon_year)
                    
                    if not success:
                        st.error(f"Decomposition failed: {results_or_error}")
                    else:
                        results_betas = results_or_error
                        
                        # Load matrix
                        file_mgr = WaveletFileManager(
                            region=st.session_state['country_name'],
                            wl_shape=st.session_state['wavelet_shape']
                        )
                        matrix_file = file_mgr.get_matrix_path(recon_year)
                        matrix = sparse.load_npz(matrix_file)
                        
                        reconstructed_signal = reconstruct(
                            time_scales=st.session_state['time_scales'],
                            reconstructed_time_scales=selected_recon_scales,
                            matrix=matrix,
                            beta_sheet=results_betas[recon_year],
                            title=f'Reconstruction',
                            xmin=0,
                            xmax=st.session_state['dpy'],
                            dpy=st.session_state['dpy'],
                            dpd=st.session_state['ndpd'],
                            add_offset=add_offset,
                            plot=False
                        )
                        
                        if reconstructed_signal is not None and len(reconstructed_signal) > 0:
                            time_axis = np.linspace(0, st.session_state['dpy'], len(reconstructed_signal))
                            
                            # Create Plotly figure
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=time_axis, y=reconstructed_signal,
                                mode='lines', name='Reconstructed',
                                line=dict(color='#2E86AB', width=1.5)
                            ))
                            
                            scales_str = ', '.join([f"{ts}h" for ts in selected_recon_scales[:5]])
                            if len(selected_recon_scales) > 5:
                                scales_str += f" ... (+{len(selected_recon_scales)-5} more)"
                            
                            title_text = f"{recon_signal} - {recon_year} - {len(selected_recon_scales)} scales"
                            
                            fig.update_layout(
                                title=title_text,
                                xaxis_title='Time (days)',
                                yaxis_title='Amplitude',
                                height=400,
                                template='plotly_white',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store for export
                            st.session_state['generated_reconstructions'].append({
                                'type': 'reconstruction',
                                'signal': recon_signal,
                                'year': recon_year,
                                'fig': fig,
                                'title': title_text,
                                'scales': selected_recon_scales,
                                'data': reconstructed_signal
                            })
                            
                            st.success("✅ Reconstruction generated and added to export list")
                        else:
                            st.error("❌ Reconstruction failed")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # Show export summary
    n_plots = len(st.session_state.get('generated_plots', []))
    n_recons = len(st.session_state.get('generated_reconstructions', []))
    if n_plots > 0 or n_recons > 0:
        st.markdown("---")
        st.info(f"📊 **Export Queue:** {n_plots} visualizations + {n_recons} reconstructions")

# ============================================================================
# STEP 5: EPN ANALYSIS
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">⚡ Step 5: EPN Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze energy storage flexibility requirements for different renewable energy mix scenarios.
    Based on **Clerjon & Perdu (2019)** methodology.
    """)
    
    # Configuration
    with st.expander("⚙️ EPN Configuration", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            epn_satisfaction_rate = st.slider(
                "Satisfaction Rate (%)", min_value=80.0, max_value=100.0, value=95.0, step=0.5,
                key="epn_satisfaction_slider"
            )
            epn_satisfactions = [epn_satisfaction_rate]
        
        with col2:
            epn_load_factor = st.number_input(
                "Load Factor (MW)", min_value=1000, max_value=100000, value=54000, step=1000,
                key="epn_load_factor_input"
            )
        
        st.markdown("**Metrics to display:**")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            epn_show_energy = st.checkbox("Energy", value=True, key="epn_show_energy")
        with mc2:
            epn_show_uf = st.checkbox("Utilization Factor", value=False, key="epn_show_uf")
        with mc3:
            epn_show_service = st.checkbox("Service", value=False, key="epn_show_service")
        
        st.markdown("---")
        epn_reference_signal = st.selectbox(
            "Reference signal for translations",
            options=['Consumption', 'PV', 'Wind'], index=0,
            key="epn_reference_signal_select"
        )
    
    # Scenario definition with coupled sliders
    st.markdown("### Energy Mix Scenarios")
    
    st.markdown("""
    **Fixed:** 🔴 100% PV | 🔵 100% Wind | ⚫ 0% ENR (dotted)
    """)
    
    st.markdown("**Custom Mix (bold line):**")
    
    # Initialize coupled sliders
    if 'epn_pv_share' not in st.session_state:
        st.session_state['epn_pv_share'] = 10
    if 'epn_wind_share' not in st.session_state:
        st.session_state['epn_wind_share'] = 10
    
    def on_pv_change():
        new_pv = st.session_state['epn_pv_widget']
        if new_pv + st.session_state['epn_wind_share'] > 100:
            st.session_state['epn_wind_share'] = 100 - new_pv
            st.session_state['epn_wind_widget'] = st.session_state['epn_wind_share']
        st.session_state['epn_pv_share'] = new_pv
    
    def on_wind_change():
        new_wind = st.session_state['epn_wind_widget']
        if st.session_state['epn_pv_share'] + new_wind > 100:
            st.session_state['epn_pv_share'] = 100 - new_wind
            st.session_state['epn_pv_widget'] = st.session_state['epn_pv_share']
        st.session_state['epn_wind_share'] = new_wind
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.slider("PV Share (%)", 0, 100, st.session_state['epn_pv_share'], 5,
                  key="epn_pv_widget", on_change=on_pv_change)
    with c2:
        st.slider("Wind Share (%)", 0, 100, st.session_state['epn_wind_share'], 5,
                  key="epn_wind_widget", on_change=on_wind_change)
    with c3:
        total_enr = st.session_state['epn_pv_share'] + st.session_state['epn_wind_share']
        st.metric("Total ENR", f"{total_enr}%")
    
    custom_pv = st.session_state['epn_pv_share']
    custom_wind = st.session_state['epn_wind_share']
    custom_mix_name = f"{custom_pv}% PV + {custom_wind}% Wind"
    
    if total_enr == 0:
        st.info(f"🟢 **Custom Mix:** {custom_mix_name} (Same as 0% ENR)")
    else:
        st.success(f"🟢 **Custom Mix:** {custom_mix_name}")
    
    # Check decompositions
    epn_year = st.session_state.get('year_to_process', 'Unknown')
    epn_time_scales = st.session_state.get('time_scales', [])
    epn_dpy = st.session_state.get('dpy', 365)
    
    epn_required = ['Consumption', 'PV', 'Wind']
    epn_all_decomps = st.session_state.get('all_decompositions', {})
    
    epn_missing = [sig for sig in epn_required if f"{sig}_{epn_year}" not in epn_all_decomps]
    
    if epn_missing:
        st.warning(f"⚠️ Missing: {', '.join(epn_missing)}")
        
        if st.button("🔄 Decompose Missing Signals", key="epn_decompose"):
            with st.spinner("Decomposing..."):
                for sig in epn_missing:
                    success, _ = ensure_decomposition(sig, epn_year)
                    if not success:
                        st.error(f"Failed to decompose {sig}")
                st.rerun()
    else:
        # Compute EPN
        betas_Load = epn_all_decomps[f'Consumption_{epn_year}']['results_betas']
        betas_PV = epn_all_decomps[f'PV_{epn_year}']['results_betas']
        betas_Wind = epn_all_decomps[f'Wind_{epn_year}']['results_betas']
        
        scenarios = [
            {'name': '100% PV', 'pv': 1.0, 'wind': 0.0},
            {'name': '100% Wind', 'pv': 0.0, 'wind': 1.0},
            {'name': '0% ENR', 'pv': 0.0, 'wind': 0.0},
            {'name': custom_mix_name, 'pv': custom_pv/100, 'wind': custom_wind/100},
        ]
        
        epn_Emax, epn_UF, epn_Serv, epn_Pmax, epn_names = [], [], [], [], []
        
        for scen in scenarios:
            pmc = [
                scen['pv'] * np.array(betas_PV[epn_year][i]) + 
                scen['wind'] * np.array(betas_Wind[epn_year][i]) - 
                np.array(betas_Load[epn_year][i]) 
                for i in range(len(epn_time_scales))
            ]
            
            result = calc_epn(pmc, epn_satisfactions, epn_time_scales, epn_dpy, epn_load_factor, shape='square')
            epn_Emax.append(result['emax'])
            epn_UF.append(result['uf'])
            epn_Serv.append(result['serv'])
            epn_Pmax.append(result['pmax'])
            epn_names.append(scen['name'])
        
        # Display
        st.markdown("---")
        st.markdown("### 📊 EPN Results")
        
        disp_metrics = []
        if epn_show_energy: disp_metrics.append('energy')
        if epn_show_uf: disp_metrics.append('uf')
        if epn_show_service: disp_metrics.append('service')
        
        if not disp_metrics:
            st.warning("Select at least one metric")
        else:
            epn_colors = ['#EE7733', '#0077BB', '#333333', '#009988']
            epn_styles = ['solid', 'solid', 'dot', 'solid']
            epn_widths = [3, 3, 3, 5]
            
            figures = plot_EPN_scenarios_plotly(
                Emax=epn_Emax, UF=epn_UF, Serv=epn_Serv,
                time_scales=epn_time_scales,
                scenario_names_list=epn_names,
                satisfactions=epn_satisfactions,
                title=f"{epn_year}",
                metrics=disp_metrics,
                colors=epn_colors,
                line_styles=epn_styles,
                line_widths=epn_widths,
                height=650,
                show_plots=False
            )
            
            for metric, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True, key=f"epn_{metric}_chart")
            
            # Store for export
            st.session_state['epn_figures'] = figures
            st.session_state['epn_params'] = {
                'year': epn_year,
                'satisfaction': epn_satisfaction_rate,
                'load_factor': epn_load_factor,
                'scenarios': epn_names,
                'custom_mix': custom_mix_name
            }

# ============================================================================
# EXPORT: HTML REPORT WITH INTERACTIVE PLOTS
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">📥 Export Report</div>', unsafe_allow_html=True)
    
    # Summary
    n_plots = len(st.session_state.get('generated_plots', []))
    n_recons = len(st.session_state.get('generated_reconstructions', []))
    has_epn = 'epn_figures' in st.session_state
    
    st.markdown(f"""
    **Export Content:**
    - 📊 Visualizations (heatmap, FFT): {n_plots}
    - 🔄 Reconstructions: {n_recons}
    - ⚡ EPN Analysis: {'Yes' if has_epn else 'No'}
    """)
    
    default_filename = f"wavelet_report_{st.session_state.get('country_name', 'region')}_{st.session_state.get('year_to_process', 'year')}"
    export_filename = st.text_input("Filename", value=default_filename)
    
    if st.button("📄 Generate HTML Report", type="primary"):
        with st.spinner("Generating HTML report..."):
            try:
                import base64
                
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{export_filename}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #A23B72; margin-top: 40px; }}
        .info-box {{ background: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #2E86AB; margin: 15px 0; }}
        .plot-container {{ margin: 30px 0; }}
        .metadata {{ background: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 0.9em; margin-top: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #2E86AB; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>📊 Wavelet Decomposition Analysis Report</h1>
    
    <div class="info-box">
        <strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>Region:</strong> {st.session_state.get('country_name', 'N/A')}<br>
        <strong>Signal:</strong> {st.session_state.get('signal_type', 'N/A')}<br>
        <strong>Year:</strong> {st.session_state.get('year_to_process', 'N/A')}<br>
        <strong>Wavelet Shape:</strong> {st.session_state.get('wavelet_shape', 'N/A')}
    </div>
    
    <h2>Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Data Points/Day</td><td>{st.session_state.get('ndpd', 'N/A')}</td></tr>
        <tr><td>Days/Year</td><td>{st.session_state.get('dpy', 'N/A')}</td></tr>
        <tr><td>Time Scales</td><td>{', '.join([f'{ts}h' for ts in st.session_state.get('time_scales', [])])}</td></tr>
    </table>
"""
                
                # Add matplotlib figures as images
                if n_plots > 0:
                    html_content += "<h2>Visualizations</h2>\n"
                    
                    for i, plot_data in enumerate(st.session_state['generated_plots']):
                        if plot_data.get('fig') is not None:
                            buf = BytesIO()
                            plot_data['fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            img_b64 = base64.b64encode(buf.read()).decode()
                            buf.close()
                            
                            html_content += f"""
    <div class="plot-container">
        <h3>{plot_data.get('title', f'Plot {i+1}')}</h3>
        <img src="data:image/png;base64,{img_b64}" style="max-width:100%;">
        <div class="metadata">
            <strong>Type:</strong> {plot_data.get('type', 'N/A')} | 
            <strong>Signal:</strong> {plot_data.get('signal', 'N/A')} | 
            <strong>Year:</strong> {plot_data.get('year', 'N/A')}
        </div>
    </div>
"""
                
                # Add Plotly reconstructions as interactive
                if n_recons > 0:
                    html_content += "<h2>Reconstructions (Interactive)</h2>\n"
                    
                    for i, recon_data in enumerate(st.session_state['generated_reconstructions']):
                        if recon_data.get('fig') is not None:
                            fig_html = recon_data['fig'].to_html(full_html=False, include_plotlyjs=False)
                            
                            html_content += f"""
    <div class="plot-container">
        <h3>{recon_data.get('title', f'Reconstruction {i+1}')}</h3>
        {fig_html}
        <div class="metadata">
            <strong>Signal:</strong> {recon_data.get('signal', 'N/A')} | 
            <strong>Year:</strong> {recon_data.get('year', 'N/A')} |
            <strong>Scales:</strong> {len(recon_data.get('scales', []))}
        </div>
    </div>
"""
                
                # Add EPN figures as interactive
                if has_epn:
                    html_content += "<h2>EPN Analysis (Interactive)</h2>\n"
                    
                    params = st.session_state.get('epn_params', {})
                    html_content += f"""
    <div class="info-box">
        <strong>Year:</strong> {params.get('year', 'N/A')} | 
        <strong>Satisfaction:</strong> {params.get('satisfaction', 'N/A')}% |
        <strong>Load Factor:</strong> {params.get('load_factor', 'N/A')} MW
    </div>
"""
                    
                    for metric, fig in st.session_state['epn_figures'].items():
                        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
                        html_content += f"""
    <div class="plot-container">
        <h3>EPN - {metric.capitalize()}</h3>
        {fig_html}
    </div>
"""
                
                # Close HTML
                html_content += """
    <hr>
    <p style="text-align:center; color:gray;">
        Generated with Wavelet Decomposition Analysis Interface<br>
        Based on Clerjon & Perdu (2019) methodology
    </p>
</body>
</html>
"""
                
                # Download button
                st.download_button(
                    label="⬇️ Download HTML Report",
                    data=html_content,
                    file_name=f"{export_filename}.html",
                    mime="text/html"
                )
                
                st.success(f"✅ Report ready! Contains {n_plots} plots, {n_recons} reconstructions" + 
                          (", EPN analysis" if has_epn else ""))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Clear export queue
    if st.button("🗑️ Clear Export Queue"):
        st.session_state['generated_plots'] = []
        st.session_state['generated_reconstructions'] = []
        if 'epn_figures' in st.session_state:
            del st.session_state['epn_figures']
        st.success("Export queue cleared")
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
📊 Wavelet Decomposition Analysis | Clerjon & Perdu (2019)
</div>
""", unsafe_allow_html=True)