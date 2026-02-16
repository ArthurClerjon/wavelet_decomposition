"""
Wavelet Decomposition Analysis Interface
==========================================
Interactive Streamlit app for analyzing time series using wavelet decomposition.

Based on the Clerjon & Perdu (2019) methodology.

VERSION V3:
- Step 1: Upload Data
- Step 2: Signal Selection & Visualization (multi-signal, subplots)
- Step 3: Run Decomposition
- Step 4: Analysis (Heatmap/FFT with unified subplot system)
- Step 5: Reconstruction (separate step with subplots)
- Step 6: EPN Analysis (fixed coupled sliders)
- Export
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE AND DESCRIPTION
# ============================================================================

st.markdown('<div class="main-header">📊 Wavelet Decomposition Analysis</div>', unsafe_allow_html=True)

st.markdown("""
**Workflow:**
1. 📁 Upload Excel file
2. 🎯 Select & visualize signals
3. 🚀 Run wavelet decomposition
4. 📈 Analysis (Heatmap/FFT subplots)
5. 🔄 Reconstruction (subplots)
6. ⚡ EPN Analysis
7. 📥 Export HTML report
""")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("## 🎛️ Configuration")

wavelet_shape = st.sidebar.radio(
    "Wavelet shape",
    options=['square', 'sine'],
    index=0
)

with st.sidebar.expander("ℹ️ About Wavelet Decomposition", expanded=False):
    st.markdown("""
    ### 15 Time Scales
    
    **Daily:** 0.75h, 1.5h, 3h, 6h, 12h, 24h
    
    **Weekly:** 42h, 84h, 168h
    
    **Yearly:** 273.75h, 547.5h, 1095h, 2190h, 4380h, 8760h
    
    Based on **Clerjon & Perdu (2019)**
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

# Time scale labels
TIME_SCALE_LABELS = {
    0.75: "0.75h", 1.5: "1.5h", 3.0: "3h", 6.0: "6h", 12.0: "12h", 24.0: "24h",
    42.0: "42h", 84.0: "84h", 168.0: "168h", 273.75: "274h", 547.5: "548h",
    1095.0: "1095h", 2190.0: "2190h", 4380.0: "4380h", 8760.0: "8760h"
}

# ============================================================================
# HELPER FUNCTION: Ensure Signal is Decomposed
# ============================================================================

def ensure_decomposition(signal, year):
    """Auto-decompose if needed. Returns: (success, results_betas or error)"""
    key = f"{signal}_{year}"
    
    if key in st.session_state.get('all_decompositions', {}):
        return True, st.session_state['all_decompositions'][key]['results_betas']
    
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
# STEP 1: FILE UPLOAD
# ============================================================================

st.markdown('<div class="section-header">📁 Step 1: Upload Data File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File: {uploaded_file.name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dpd = st.number_input("Points/day (original)", min_value=1, max_value=96, value=48)
    with col2:
        ndpd = st.number_input("Points/day (interpolated)", min_value=1, max_value=128, value=64)
    with col3:
        dpy = st.number_input("Days/year", min_value=1, max_value=366, value=365)
    
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
                st.session_state['vy'] = 6
                st.session_state['vw'] = 3
                st.session_state['vd'] = 6
                st.session_state['time_scales'] = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 
                                                   273.75, 547.5, 1095., 2190., 4380., 8760.]
                
                st.success(f"✅ Imported: {', '.join(years)}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# STEP 2: SIGNAL SELECTION AND VISUALIZATION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🎯 Step 2: Select and Visualize Signals</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_type = st.selectbox("Primary signal", options=['Consumption', 'Wind', 'PV'])
    with col2:
        year_to_process = st.selectbox("Year", options=st.session_state['years'])
    with col3:
        country_name = st.text_input("Region", value="France")
    
    st.session_state['signal_type'] = signal_type
    st.session_state['year_to_process'] = year_to_process
    st.session_state['country_name'] = country_name
    st.session_state['wavelet_shape'] = wavelet_shape
    
    # Subplot visualization
    st.markdown("### 📊 Time Series Visualization")
    
    if 'viz_subplots' not in st.session_state:
        st.session_state['viz_subplots'] = [
            {'signals': ['Consumption'], 'years': [year_to_process], 'row': 0, 'col': 0}
        ]
    
    for idx, subplot in enumerate(st.session_state['viz_subplots']):
        with st.expander(f"Subplot {idx + 1}", expanded=(idx == 0)):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                subplot['signals'] = st.multiselect(
                    "Signals", ['Consumption', 'Wind', 'PV'],
                    default=subplot['signals'], key=f"viz_sig_{idx}"
                )
            with c2:
                subplot['years'] = st.multiselect(
                    "Years", st.session_state['years'],
                    default=subplot['years'], key=f"viz_yr_{idx}"
                )
            with c3:
                if idx > 0 and st.button("🗑️", key=f"viz_rm_{idx}"):
                    st.session_state['viz_subplots'].pop(idx)
                    st.rerun()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add Below"):
            max_row = max([s['row'] for s in st.session_state['viz_subplots']]) + 1
            st.session_state['viz_subplots'].append(
                {'signals': ['Consumption'], 'years': [year_to_process], 'row': max_row, 'col': 0}
            )
            st.rerun()
    with c2:
        if st.button("➕ Add Right"):
            max_col = max([s['col'] for s in st.session_state['viz_subplots']]) + 1
            st.session_state['viz_subplots'].append(
                {'signals': ['Consumption'], 'years': [year_to_process], 'row': 0, 'col': max_col}
            )
            st.rerun()
    
    if st.button("📈 Generate Visualization", type="primary"):
        try:
            max_row = max([s['row'] for s in st.session_state['viz_subplots']]) + 1
            max_col = max([s['col'] for s in st.session_state['viz_subplots']]) + 1
            
            fig = make_subplots(rows=max_row, cols=max_col, horizontal_spacing=0.08, vertical_spacing=0.12)
            
            colors = {'Consumption': '#2E86AB', 'Wind': '#A23B72', 'PV': '#F18F01'}
            styles = ['solid', 'dash', 'dot', 'dashdot']
            
            pts = st.session_state['signal_length']
            time_axis = np.linspace(0, st.session_state['dpy'], pts)
            
            for subplot in st.session_state['viz_subplots']:
                if not subplot['signals'] or not subplot['years']:
                    continue
                row, col = subplot['row'] + 1, subplot['col'] + 1
                
                for sig in subplot['signals']:
                    for yi, yr in enumerate(subplot['years']):
                        y_idx = st.session_state['years'].index(yr)
                        data = st.session_state['stacked_input_data'][sig][y_idx*pts:(y_idx+1)*pts]
                        
                        fig.add_trace(go.Scatter(
                            x=time_axis, y=data, mode='lines', name=f"{sig} ({yr})",
                            line=dict(color=colors.get(sig, '#333'), dash=styles[yi % len(styles)])
                        ), row=row, col=col)
                
                fig.update_xaxes(title_text="Time (days)", row=row, col=col)
                fig.update_yaxes(title_text="Power", row=row, col=col)
            
            fig.update_layout(height=400*max_row, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ {str(e)}")

# ============================================================================
# STEP 3: DECOMPOSITION
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">🚀 Step 3: Run Wavelet Decomposition</div>', unsafe_allow_html=True)
    
    st.markdown(f"**Config:** 15 scales | {wavelet_shape} | {st.session_state['ndpd']} pts/day")
    
    recompute = st.checkbox("Recompute translations", value=False)
    
    if st.button("🚀 Run Decomposition", type="primary"):
        with st.spinner(f"Decomposing {signal_type} for {year_to_process}..."):
            try:
                y_idx = st.session_state['years'].index(year_to_process)
                pts = st.session_state['signal_length']
                TS = st.session_state['stacked_input_data'][signal_type][y_idx*pts:(y_idx+1)*pts]
                
                trans_file, matrix_files, results_betas, trans = wavelet_decomposition_single_TS(
                    TS, year=year_to_process, multi_year=None, country_name=country_name,
                    signal_type=signal_type, wl_shape=wavelet_shape, recompute_translation=recompute,
                    dpd=st.session_state['dpd'], ndpd=st.session_state['ndpd'],
                    vy=st.session_state['vy'], vw=st.session_state['vw'], vd=st.session_state['vd']
                )
                
                st.session_state['decomposition_done'] = True
                st.session_state['results_betas'] = results_betas
                st.session_state['trans'] = trans
                
                key = f"{signal_type}_{year_to_process}"
                st.session_state['all_decompositions'][key] = {
                    'results_betas': results_betas, 'trans_file': trans_file,
                    'matrix_files': matrix_files, 'trans': trans
                }
                
                st.success("✅ Decomposition complete!")
                
            except Exception as e:
                st.error(f"❌ {str(e)}")

# ============================================================================
# STEP 4: ANALYSIS (Heatmap/FFT with unified subplot system)
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">📈 Step 4: Analysis (Heatmap / FFT)</div>', unsafe_allow_html=True)
    
    st.markdown("Create subplots with heatmaps and/or FFT side by side.")
    
    # Initialize analysis subplots
    if 'analysis_subplots' not in st.session_state:
        st.session_state['analysis_subplots'] = []
    
    # Add subplot buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add Heatmap", key="add_hm"):
            st.session_state['analysis_subplots'].append({
                'type': 'heatmap', 'signal': 'Consumption', 'year': year_to_process,
                'scales': {ts: True for ts in st.session_state['time_scales']},
                'row': len(st.session_state['analysis_subplots']) // 2,
                'col': len(st.session_state['analysis_subplots']) % 2
            })
            st.rerun()
    with c2:
        if st.button("➕ Add FFT", key="add_fft"):
            st.session_state['analysis_subplots'].append({
                'type': 'fft', 'signal': 'Consumption', 'year': year_to_process,
                'row': len(st.session_state['analysis_subplots']) // 2,
                'col': len(st.session_state['analysis_subplots']) % 2
            })
            st.rerun()
    with c3:
        if st.session_state['analysis_subplots'] and st.button("🗑️ Clear All", key="clear_analysis"):
            st.session_state['analysis_subplots'] = []
            st.rerun()
    
    if st.session_state['analysis_subplots']:
        st.info(f"{len(st.session_state['analysis_subplots'])} subplot(s) configured")
    
    # Configure each subplot
    for idx, subplot in enumerate(st.session_state['analysis_subplots']):
        with st.expander(f"Subplot {idx+1}: {subplot['type'].upper()}", expanded=True):
            c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
            
            with c1:
                subplot['signal'] = st.selectbox(
                    "Signal", ['Consumption', 'Wind', 'PV'],
                    index=['Consumption', 'Wind', 'PV'].index(subplot['signal']),
                    key=f"an_sig_{idx}"
                )
            with c2:
                subplot['year'] = st.selectbox(
                    "Year", st.session_state['years'],
                    index=st.session_state['years'].index(subplot['year']),
                    key=f"an_yr_{idx}"
                )
            with c3:
                subplot['row'] = st.number_input("Row", min_value=0, value=subplot['row'], key=f"an_row_{idx}")
            with c4:
                subplot['col'] = st.number_input("Col", min_value=0, value=subplot['col'], key=f"an_col_{idx}")
            
            # Remove button
            if st.button("🗑️ Remove", key=f"an_rm_{idx}"):
                st.session_state['analysis_subplots'].pop(idx)
                st.rerun()
            
            # Time scales for heatmap
            if subplot['type'] == 'heatmap':
                st.markdown("**Time Scales:**")
                cb1, cb2, _ = st.columns([1, 1, 4])
                with cb1:
                    if st.button("✅ All", key=f"an_all_{idx}"):
                        for ts in st.session_state['time_scales']:
                            subplot['scales'][ts] = True
                        st.rerun()
                with cb2:
                    if st.button("❌ None", key=f"an_none_{idx}"):
                        for ts in st.session_state['time_scales']:
                            subplot['scales'][ts] = False
                        st.rerun()
                
                cols = st.columns(15)
                for i, ts in enumerate(st.session_state['time_scales']):
                    with cols[i]:
                        cur = subplot['scales'].get(ts, True)
                        new = st.checkbox(TIME_SCALE_LABELS[ts], value=cur, key=f"an_ts_{idx}_{ts}")
                        if new != cur:
                            subplot['scales'][ts] = new
    
    # Generate all analysis plots
    if st.session_state['analysis_subplots']:
        if st.button("📊 Generate Analysis Plots", type="primary", key="gen_analysis"):
            with st.spinner("Generating..."):
                # Determine grid
                max_row = max([s['row'] for s in st.session_state['analysis_subplots']]) + 1
                max_col = max([s['col'] for s in st.session_state['analysis_subplots']]) + 1
                
                # Create columns for side-by-side display
                grid_cols = st.columns(max_col)
                
                # Group subplots by position
                for subplot in st.session_state['analysis_subplots']:
                    col_idx = subplot['col']
                    
                    with grid_cols[col_idx]:
                        try:
                            if subplot['type'] == 'heatmap':
                                selected_scales = [ts for ts in st.session_state['time_scales'] 
                                                  if subplot['scales'].get(ts, True)]
                                if not selected_scales:
                                    st.warning("No scales selected")
                                    continue
                                
                                success, result = ensure_decomposition(subplot['signal'], subplot['year'])
                                if not success:
                                    st.error(result)
                                    continue
                                
                                fig = plot_betas_heatmap(
                                    results_betas=result,
                                    country_name=st.session_state['country_name'],
                                    signal_type=subplot['signal'],
                                    vy=st.session_state['vy'], vw=st.session_state['vw'], vd=st.session_state['vd'],
                                    ndpd=st.session_state['ndpd'], dpy=st.session_state['dpy'],
                                    year=subplot['year'], years=[subplot['year']],
                                    time_scales=st.session_state['time_scales'],
                                    reconstructed_time_scales=selected_scales,
                                    cmin=-0.1, cmax=0.1, ccenter=None,
                                    wl_shape=st.session_state['wavelet_shape']
                                )
                                
                                st.pyplot(fig)
                                st.caption(f"Heatmap: {subplot['signal']} - {subplot['year']}")
                                
                                st.session_state['generated_plots'].append({
                                    'type': 'heatmap', 'signal': subplot['signal'],
                                    'year': subplot['year'], 'fig': fig,
                                    'title': f"Heatmap: {subplot['signal']} - {subplot['year']}"
                                })
                            
                            elif subplot['type'] == 'fft':
                                y_idx = st.session_state['years'].index(subplot['year'])
                                pts = st.session_state['signal_length']
                                data = st.session_state['stacked_input_data'][subplot['signal']][y_idx*pts:(y_idx+1)*pts]
                                
                                fig = fft(
                                    ndpd=st.session_state['ndpd'], dpy=st.session_state['dpy'],
                                    signal_type=subplot['signal'], year=subplot['year'], input_data=data
                                )
                                
                                st.pyplot(fig)
                                st.caption(f"FFT: {subplot['signal']} - {subplot['year']}")
                                
                                st.session_state['generated_plots'].append({
                                    'type': 'fft', 'signal': subplot['signal'],
                                    'year': subplot['year'], 'fig': fig,
                                    'title': f"FFT: {subplot['signal']} - {subplot['year']}"
                                })
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                st.success("✅ Analysis plots generated and added to export queue")

# ============================================================================
# STEP 5: RECONSTRUCTION (with subplots)
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">🔄 Step 5: Signal Reconstruction</div>', unsafe_allow_html=True)
    
    st.markdown("Reconstruct signals with selected time scales. Display as subplots.")
    
    # Initialize reconstruction subplots
    if 'recon_subplots' not in st.session_state:
        st.session_state['recon_subplots'] = []
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add Reconstruction", key="add_recon"):
            st.session_state['recon_subplots'].append({
                'signal': 'Consumption', 'year': year_to_process,
                'scales': {ts: True for ts in st.session_state['time_scales']},
                'add_offset': False,
                'row': len(st.session_state['recon_subplots']) // 2,
                'col': len(st.session_state['recon_subplots']) % 2
            })
            st.rerun()
    with c2:
        if st.session_state['recon_subplots']:
            st.info(f"{len(st.session_state['recon_subplots'])} configured")
    with c3:
        if st.session_state['recon_subplots'] and st.button("🗑️ Clear All", key="clear_recon"):
            st.session_state['recon_subplots'] = []
            st.rerun()
    
    # Configure each reconstruction
    for idx, recon in enumerate(st.session_state['recon_subplots']):
        with st.expander(f"Reconstruction {idx+1}", expanded=True):
            c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
            
            with c1:
                recon['signal'] = st.selectbox(
                    "Signal", ['Consumption', 'Wind', 'PV'],
                    index=['Consumption', 'Wind', 'PV'].index(recon['signal']),
                    key=f"rec_sig_{idx}"
                )
            with c2:
                recon['year'] = st.selectbox(
                    "Year", st.session_state['years'],
                    index=st.session_state['years'].index(recon['year']),
                    key=f"rec_yr_{idx}"
                )
            with c3:
                recon['row'] = st.number_input("Row", min_value=0, value=recon['row'], key=f"rec_row_{idx}")
            with c4:
                recon['col'] = st.number_input("Col", min_value=0, value=recon['col'], key=f"rec_col_{idx}")
            
            if st.button("🗑️ Remove", key=f"rec_rm_{idx}"):
                st.session_state['recon_subplots'].pop(idx)
                st.rerun()
            
            # Time scales
            st.markdown("**Time Scales:**")
            cb1, cb2, _ = st.columns([1, 1, 4])
            with cb1:
                if st.button("✅ All", key=f"rec_all_{idx}"):
                    for ts in st.session_state['time_scales']:
                        recon['scales'][ts] = True
                    st.rerun()
            with cb2:
                if st.button("❌ None", key=f"rec_none_{idx}"):
                    for ts in st.session_state['time_scales']:
                        recon['scales'][ts] = False
                    st.rerun()
            
            cols = st.columns(15)
            for i, ts in enumerate(st.session_state['time_scales']):
                with cols[i]:
                    cur = recon['scales'].get(ts, True)
                    new = st.checkbox(TIME_SCALE_LABELS[ts], value=cur, key=f"rec_ts_{idx}_{ts}")
                    if new != cur:
                        recon['scales'][ts] = new
            
            recon['add_offset'] = st.checkbox("Add offset (DC)", value=recon['add_offset'], key=f"rec_off_{idx}")
    
    # Generate all reconstructions
    if st.session_state['recon_subplots']:
        if st.button("🔄 Generate Reconstructions", type="primary", key="gen_recon"):
            with st.spinner("Generating..."):
                max_col = max([r['col'] for r in st.session_state['recon_subplots']]) + 1
                grid_cols = st.columns(max_col)
                
                for recon in st.session_state['recon_subplots']:
                    col_idx = recon['col']
                    
                    with grid_cols[col_idx]:
                        try:
                            selected_scales = [ts for ts in st.session_state['time_scales'] 
                                              if recon['scales'].get(ts, True)]
                            if not selected_scales:
                                st.warning("No scales selected")
                                continue
                            
                            success, result = ensure_decomposition(recon['signal'], recon['year'])
                            if not success:
                                st.error(result)
                                continue
                            
                            file_mgr = WaveletFileManager(
                                region=st.session_state['country_name'],
                                wl_shape=st.session_state['wavelet_shape']
                            )
                            matrix = sparse.load_npz(file_mgr.get_matrix_path(recon['year']))
                            
                            recon_signal = reconstruct(
                                time_scales=st.session_state['time_scales'],
                                reconstructed_time_scales=selected_scales,
                                matrix=matrix,
                                beta_sheet=result[recon['year']],
                                title='', xmin=0, xmax=st.session_state['dpy'],
                                dpy=st.session_state['dpy'], dpd=st.session_state['ndpd'],
                                add_offset=recon['add_offset'], plot=False
                            )
                            
                            if recon_signal is not None:
                                time_axis = np.linspace(0, st.session_state['dpy'], len(recon_signal))
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=time_axis, y=recon_signal, mode='lines', name='Reconstructed',
                                    line=dict(color='#2E86AB', width=1.5)
                                ))
                                
                                title = f"{recon['signal']} - {recon['year']} ({len(selected_scales)} scales)"
                                fig.update_layout(
                                    title=title, xaxis_title='Time (days)', yaxis_title='Amplitude',
                                    height=350, template='plotly_white', hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"recon_fig_{recon['signal']}_{recon['year']}_{col_idx}")
                                
                                st.session_state['generated_reconstructions'].append({
                                    'signal': recon['signal'], 'year': recon['year'],
                                    'fig': fig, 'title': title, 'scales': selected_scales
                                })
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                st.success("✅ Reconstructions generated and added to export queue")

# ============================================================================
# STEP 6: EPN ANALYSIS (Fixed coupled sliders)
# ============================================================================

if 'decomposition_done' in st.session_state and st.session_state['decomposition_done']:
    
    st.markdown('<div class="section-header">⚡ Step 6: EPN Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("Energy storage flexibility requirements based on **Clerjon & Perdu (2019)**.")
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            epn_satisfaction = st.slider("Satisfaction (%)", 80.0, 100.0, 95.0, 0.5, key="epn_sat")
        with c2:
            epn_load = st.number_input("Load Factor (MW)", 1000, 100000, 54000, 1000, key="epn_load")
        
        st.markdown("**Metrics:**")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            show_energy = st.checkbox("Energy", True, key="epn_energy")
        with mc2:
            show_uf = st.checkbox("Utilization Factor", False, key="epn_uf")
        with mc3:
            show_service = st.checkbox("Service", False, key="epn_service")
    
    # Scenarios with FIXED coupled sliders
    st.markdown("### Energy Mix Scenarios")
    st.markdown("**Fixed:** 🔴 100% PV | 🔵 100% Wind | ⚫ 0% ENR (dotted)")
    st.markdown("**Custom Mix (bold line):**")
    
    # Initialize slider values
    if 'epn_pv' not in st.session_state:
        st.session_state['epn_pv'] = 10
    if 'epn_wind' not in st.session_state:
        st.session_state['epn_wind'] = 10
    
    # FIXED CALLBACKS - each slider adjusts the other when needed
    def on_pv_change():
        """PV is being moved - adjust Wind if total > 100%"""
        new_pv = st.session_state['pv_slider']
        current_wind = st.session_state['epn_wind']
        
        if new_pv + current_wind > 100:
            # Wind must decrease
            st.session_state['epn_wind'] = 100 - new_pv
        
        st.session_state['epn_pv'] = new_pv
    
    def on_wind_change():
        """Wind is being moved - adjust PV if total > 100%"""
        new_wind = st.session_state['wind_slider']
        current_pv = st.session_state['epn_pv']
        
        if current_pv + new_wind > 100:
            # PV must decrease
            st.session_state['epn_pv'] = 100 - new_wind
        
        st.session_state['epn_wind'] = new_wind
    
    c1, c2, c3 = st.columns([2, 2, 1])
    
    with c1:
        st.slider(
            "PV Share (%)", 0, 100, 
            st.session_state['epn_pv'], 5,
            key="pv_slider",
            on_change=on_pv_change
        )
    
    with c2:
        st.slider(
            "Wind Share (%)", 0, 100,
            st.session_state['epn_wind'], 5,
            key="wind_slider",
            on_change=on_wind_change
        )
    
    with c3:
        total = st.session_state['epn_pv'] + st.session_state['epn_wind']
        st.metric("Total ENR", f"{total}%")
    
    pv_share = st.session_state['epn_pv']
    wind_share = st.session_state['epn_wind']
    custom_name = f"{pv_share}% PV + {wind_share}% Wind"
    
    if total == 0:
        st.info(f"🟢 Custom Mix: {custom_name} (Same as 0% ENR)")
    else:
        st.success(f"🟢 Custom Mix: {custom_name}")
    
    # Check decompositions
    epn_year = st.session_state['year_to_process']
    required = ['Consumption', 'PV', 'Wind']
    missing = [s for s in required if f"{s}_{epn_year}" not in st.session_state['all_decompositions']]
    
    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)}")
        if st.button("🔄 Decompose Missing", key="epn_decomp"):
            for sig in missing:
                ensure_decomposition(sig, epn_year)
            st.rerun()
    else:
        # Compute EPN
        decomps = st.session_state['all_decompositions']
        betas_Load = decomps[f'Consumption_{epn_year}']['results_betas']
        betas_PV = decomps[f'PV_{epn_year}']['results_betas']
        betas_Wind = decomps[f'Wind_{epn_year}']['results_betas']
        
        time_scales = st.session_state['time_scales']
        
        scenarios = [
            {'name': '100% PV', 'pv': 1.0, 'wind': 0.0},
            {'name': '100% Wind', 'pv': 0.0, 'wind': 1.0},
            {'name': '0% ENR', 'pv': 0.0, 'wind': 0.0},
            {'name': custom_name, 'pv': pv_share/100, 'wind': wind_share/100},
        ]
        
        Emax, UF, Serv, Pmax, names = [], [], [], [], []
        
        for scen in scenarios:
            pmc = [
                scen['pv'] * np.array(betas_PV[epn_year][i]) +
                scen['wind'] * np.array(betas_Wind[epn_year][i]) -
                np.array(betas_Load[epn_year][i])
                for i in range(len(time_scales))
            ]
            
            result = calc_epn(pmc, [epn_satisfaction], time_scales, st.session_state['dpy'], epn_load, shape='square')
            Emax.append(result['emax'])
            UF.append(result['uf'])
            Serv.append(result['serv'])
            Pmax.append(result['pmax'])
            names.append(scen['name'])
        
        # Display
        st.markdown("---")
        st.markdown("### 📊 EPN Results")
        
        metrics = []
        if show_energy: metrics.append('energy')
        if show_uf: metrics.append('uf')
        if show_service: metrics.append('service')
        
        if not metrics:
            st.warning("Select at least one metric")
        else:
            figures = plot_EPN_scenarios_plotly(
                Emax=Emax, UF=UF, Serv=Serv,
                time_scales=time_scales,
                scenario_names_list=names,
                satisfactions=[epn_satisfaction],
                title=epn_year,
                metrics=metrics,
                colors=['#EE7733', '#0077BB', '#333333', '#009988'],
                line_styles=['solid', 'solid', 'dot', 'solid'],
                line_widths=[3, 3, 3, 5],
                height=650,
                show_plots=False
            )
            
            # FIXED: Remove key or make it unique with timestamp
            for metric, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True)
            
            st.session_state['epn_figures'] = figures
            st.session_state['epn_params'] = {
                'year': epn_year, 'satisfaction': epn_satisfaction,
                'load_factor': epn_load, 'custom_mix': custom_name
            }

# ============================================================================
# EXPORT
# ============================================================================

if 'data_imported' in st.session_state and st.session_state['data_imported']:
    
    st.markdown('<div class="section-header">📥 Export Report</div>', unsafe_allow_html=True)
    
    n_plots = len(st.session_state.get('generated_plots', []))
    n_recons = len(st.session_state.get('generated_reconstructions', []))
    has_epn = 'epn_figures' in st.session_state
    
    st.markdown(f"**Content:** {n_plots} plots | {n_recons} reconstructions | EPN: {'Yes' if has_epn else 'No'}")
    
    filename = st.text_input("Filename", value=f"report_{st.session_state.get('country_name', 'region')}")
    
    if st.button("📄 Generate HTML", type="primary"):
        with st.spinner("Generating..."):
            try:
                import base64
                
                html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{filename}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{{font-family:Arial;max-width:1400px;margin:0 auto;padding:20px}}
h1{{color:#2E86AB;border-bottom:3px solid #2E86AB}}
h2{{color:#A23B72;margin-top:40px}}
.info{{background:#f0f2f6;padding:15px;border-radius:5px;border-left:4px solid #2E86AB;margin:15px 0}}
.plot{{margin:30px 0}}
</style></head><body>
<h1>📊 Wavelet Analysis Report</h1>
<div class="info">
<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}<br>
<b>Region:</b> {st.session_state.get('country_name','N/A')}<br>
<b>Year:</b> {st.session_state.get('year_to_process','N/A')}
</div>"""
                
                if n_plots > 0:
                    html += "<h2>Analysis</h2>"
                    for p in st.session_state['generated_plots']:
                        if p.get('fig'):
                            buf = BytesIO()
                            p['fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            b64 = base64.b64encode(buf.read()).decode()
                            html += f'<div class="plot"><h3>{p.get("title","")}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"></div>'
                
                if n_recons > 0:
                    html += "<h2>Reconstructions</h2>"
                    for r in st.session_state['generated_reconstructions']:
                        if r.get('fig'):
                            html += f'<div class="plot"><h3>{r.get("title","")}</h3>{r["fig"].to_html(full_html=False, include_plotlyjs=False)}</div>'
                
                if has_epn:
                    html += "<h2>EPN Analysis</h2>"
                    params = st.session_state.get('epn_params', {})
                    html += f'<div class="info"><b>Year:</b> {params.get("year")} | <b>Satisfaction:</b> {params.get("satisfaction")}% | <b>Mix:</b> {params.get("custom_mix")}</div>'
                    for m, f in st.session_state['epn_figures'].items():
                        html += f'<div class="plot"><h3>EPN - {m.capitalize()}</h3>{f.to_html(full_html=False, include_plotlyjs=False)}</div>'
                
                html += "<hr><p style='text-align:center;color:gray'>Wavelet Analysis | Clerjon & Perdu (2019)</p></body></html>"
                
                st.download_button("⬇️ Download", html, f"{filename}.html", "text/html")
                st.success("✅ Ready!")
                
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    if st.button("🗑️ Clear Queue"):
        st.session_state['generated_plots'] = []
        st.session_state['generated_reconstructions'] = []
        if 'epn_figures' in st.session_state:
            del st.session_state['epn_figures']
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray'>📊 Wavelet Analysis | Clerjon & Perdu (2019)</div>", unsafe_allow_html=True)