"""
Plotting functions for wavelet decomposition analysis
Includes heatmaps, FFT plots, and EPN visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import os
import glob
import pickle as pkl
from scipy import sparse
from scipy.sparse.linalg import lsqr
import scipy.fftpack
import xlsxwriter
import matplotlib.ticker as mticke
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import required modules
import config
from wavelet_decomposition import reconstruct
from file_manager import WaveletFileManager

def plot_betas_heatmap(
    results_betas: dict,
    country_name: str,
    signal_type: str,
    vy: int,
    vw: int,
    vd: int,
    ndpd: int,
    dpy: int,
    year: str,
    years: list,
    time_scales: list,
    reconstructed_time_scales: list = None,
    cmin: float = None,
    cmax: float = None,
    ccenter: float = None,
    wl_shape: str = 'square',
    base_results_dir: str = 'results'
) -> None:
    """
    Plot wavelet coefficients heatmap for selected time scales.
    
    This function reconstructs ONLY the selected time scales (not all 15),
    making it ~5x faster than preprocessing all scales.
    
    Args:
        results_betas: Dictionary containing wavelet coefficients for each year
        country_name: Region/country name (e.g., 'France', 'Germany')
        signal_type: Type of signal ('Consumption', 'Wind', or 'PV')
        vy: Number of yearly wavelets (must match decomposition parameters)
        vw: Number of weekly wavelets (must match decomposition parameters)
        vd: Number of daily wavelets (must match decomposition parameters)
        ndpd: Number of data points per day
        dpy: Days per year
        year: Year of the data to plot
        years: List of all years in the dataset
        time_scales: List of all available time scales
        reconstructed_time_scales: List of time scales to plot 
                                   (default: [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380.,
        8760.] = day, week, year)
        cmin: Minimum color value (default: None = auto)
        cmax: Maximum color value (default: None = auto)
        ccenter: Center color value (default: None = auto center at 0)
        wl_shape: Wavelet shape used in decomposition ('square' or 'sine')
                  Used to locate correct matrix directory
        base_results_dir: Base directory for results (default: 'results')
    """
    
    # =========================================================================
    # 1. SMART DEFAULTS
    # =========================================================================
    
    # Default to showing day, week, and year scales (most common case)
    if reconstructed_time_scales is None:
        reconstructed_time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380.,
        8760.]
        print(f"Using default time scales")
    
    # =========================================================================
    # 2. DETERMINE MATRIX PATH USING FileManager
    # =========================================================================
    
    # Initialize FileManager with wl_shape (controls path: results/{region}/{wl_shape}/)
    file_mgr = WaveletFileManager(
        base_dir=base_results_dir, 
        region=country_name,
        wl_shape=wl_shape
    )
    
    # Get matrix file path (FileManager handles wl_shape internally)
    matrix_file = file_mgr.get_matrix_path(year)
    
    print(f"Looking for matrix: {matrix_file}")
    if not os.path.exists(matrix_file):
        # Provide helpful error message
        raise FileNotFoundError(
            f"Matrix file not found: {matrix_file}\n"
            f"Expected in: {os.path.dirname(matrix_file)}\n"
            f"\nPossible causes:\n"
            f"  1. Decomposition not run for year {year}\n"
            f"  2. Wrong wl_shape (currently '{wl_shape}')\n"
            f"  3. Wrong country_name (currently '{country_name}')\n"
            f"  4. Matrix saved in different location\n"
            f"\nCheck these directories:\n"
            f"  - {os.path.join(config.MATRIX_DIR, 'square_shape')}\n"
            f"  - {os.path.join(config.MATRIX_DIR, 'sine_shape')}\n"
            f"\nRun wavelet_decomposition_single_TS() first!"
        )
    
    print(f"✅ Loading matrix from: {matrix_file}")
    matrix_sparse = sparse.load_npz(matrix_file)
    
    # Convert to dense array for reconstruction function
    matrix = sparse.csr_matrix.todense(matrix_sparse)
    matrix = np.asarray(matrix)
    print(f"✅ Matrix loaded: shape {matrix.shape}")

    # =========================================================================
    # 3. VALIDATE PARAMETERS
    # =========================================================================
    
    # Verify year exists
    if year not in years:
        raise ValueError(f"Year '{year}' not in years list: {years}")
    
    # Verify decomposition parameters match
    Nb_vec = vy + vw + vd
    max_nb_betas = dpy * ndpd
    expected_scales = Nb_vec + 1  # +1 for offset
    actual_scales = len(results_betas[year])
    
    if expected_scales != actual_scales:
        raise ValueError(
            f"Parameter mismatch!\n"
            f"Expected {expected_scales} time scales (vy={vy}, vw={vw}, vd={vd})\n"
            f"But results_betas['{year}'] has {actual_scales} time scales\n"
            f"\nMake sure vy, vw, vd match your wavelet_decomposition_single_TS() call!\n"
            f"(Default decomposition values: vy=6, vw=3, vd=6)"
        )

    # =========================================================================
    # 4. RECONSTRUCT ONLY SELECTED TIME SCALES (Speed optimization!)
    # =========================================================================
    
    print(f"Reconstructing {len(reconstructed_time_scales)} time scales "
          f"(out of {len(time_scales)} total available)...")
    
    # Create an empty DataFrame
    df = pd.DataFrame(np.nan, index=range(Nb_vec), columns=range(max_nb_betas)).transpose()

    # Fill DataFrame with reconstructed wavelet coefficients
    for k, ts in enumerate(time_scales):
        if ts in reconstructed_time_scales:
            # Reconstruct the signal for this time scale
            new_vec = reconstruct(
                time_scales,
                [ts],
                matrix,
                results_betas[year],
                f"{signal_type} signal reconstruction",
                xmin=0,
                xmax=dpy,
                dpy=dpy,
                dpd=ndpd,
                add_offset=False,
                plot=False  # Don't show intermediate plots
            )
            plt.close()  # Clean up any stray plots
            df[k] = pd.DataFrame({'betas': new_vec})

    # Filter DataFrame to only include selected time scales
    selected_indices = [i for i, ts in enumerate(time_scales) if ts in reconstructed_time_scales]
    df = df.iloc[:, selected_indices]
    
    print(f"✅ Reconstruction complete")

    # =========================================================================
    # 5. PLOT SETUP
    # =========================================================================
    
    # Plot aesthetic settings
    sns.set()
    sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")
    plt.rc('font', family='serif')

    # =========================================================================
    # 6. PREPARE TIME SCALE LABELS (Y-AXIS)
    # =========================================================================
    
    time_scale_labels = []
    y_positions = []
    for i, ts in enumerate(reconstructed_time_scales):
        if ts == 24.:
            time_scale_labels.append('day')
        elif ts == 168.:
            time_scale_labels.append('week')
        elif ts == 8760.:
            time_scale_labels.append('year')
        else:
            time_scale_labels.append(f"{ts}h")
        y_positions.append(i + 0.5)

    # =========================================================================
    # 7. PREPARE MONTH LABELS (X-AXIS)
    # =========================================================================
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Days in each month (non-leap year)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Calculate cumulative days (start of each month)
    month_start_days = [0]  # January starts at day 0
    for days in days_in_month[:-1]:  # All months except December
        month_start_days.append(month_start_days[-1] + days)
    
    # Calculate middle of each month for label positioning
    month_middle_days = []
    for i, start_day in enumerate(month_start_days):
        middle = start_day + days_in_month[i] / 2
        month_middle_days.append(middle)
    
    # Convert days to data points (multiply by ndpd)
    month_label_positions = [day * ndpd for day in month_middle_days]
    month_boundary_positions = [day * ndpd for day in month_start_days] + [365 * ndpd]

    # =========================================================================
    # 8. CREATE HEATMAP
    # =========================================================================
    
    # Figure settings
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)

    # Create heatmap
    Z = df.transpose()
    ax = sns.heatmap(
        Z,
        cmap='coolwarm',
        center=ccenter,
        vmin=cmin,
        vmax=cmax,
        cbar=False
    )

    ax.set_aspect("auto")
    plt.ylim(len(reconstructed_time_scales), 0)

    # =========================================================================
    # 9. ADD VERTICAL LINES FOR MONTH BOUNDARIES
    # =========================================================================
    
    # Add vertical lines at the start of each month
    for month_boundary in month_boundary_positions[1:-1]:  # Skip first (0) and last (end)
        ax.axvline(x=month_boundary, color='black', linestyle='--', 
                   linewidth=1.2, alpha=1., zorder=10)

    # =========================================================================
    # 10. ADD COLORBAR
    # =========================================================================
    
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_label('Charge - Discharge power', fontsize=16)

    # =========================================================================
    # 11. SET AXIS LABELS
    # =========================================================================
    
    # Set y-axis labels (time scales)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(time_scale_labels, minor=False, rotation=0)

    # Set x-axis labels (months)
    ax.set_xticks(month_label_positions)
    ax.set_xticklabels(month_names, minor=False, rotation=0)

    # =========================================================================
    # 12. SET TITLES AND LABELS
    # =========================================================================
    
    plt.ylabel('Storage time scale (hours)', fontsize=20, fontweight='normal')
    plt.xlabel('Month', fontsize=20, fontweight='normal')
    plt.title(f'Wavelet transform of the signal "{signal_type}" in {year}', 
              fontsize=20, fontweight='normal')

    plt.ylim(len(reconstructed_time_scales), 0)
    fig.tight_layout()
  
    print(f"✅ Heatmap displayed for {signal_type} - {year}")
    return fig

def fft(ndpd, dpy, signal_type, year, input_data):
    """
    Fast Fourier Transform (FFT) spectrum analysis of input time series.
    """
    # =========================================================================
    # PLOT SETUP - Matches heatmap styling
    # =========================================================================
    
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.})
    
    # Modern font settings (matches heatmap)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20

    # =========================================================================
    # FFT COMPUTATION
    # =========================================================================
    
    signal_length = dpy * ndpd
    
    # Number of sample points
    N = len(input_data)
    
    # Sample spacing
    T = 1.0 / N
    x = np.linspace(0, int(N*T), N)
    y = input_data - np.mean(input_data)
    yf = np.absolute(scipy.fftpack.fft(y))
    xf = 8760./np.linspace(0, 1.0/(2.0*T), int(N/2))

    # Reference time scales
    xcoords = [1, 12, 52, 365, 365*2, 365*24]
    ylabel = ['Year', 'Month', 'Week', 'Day', '12h', 'Hour']
    
    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    
    yf_abs = 2.0/N * np.abs(yf[:N//2])
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.set_xscale('log')
    plt.plot(yf_abs)
    plt.xlim(0.9,365*64/2)

    # plt.grid(True, which="both")
    # =========================================================================
    # LABELS AND TITLE - Matches heatmap styling
    # =========================================================================
    
    plt.ylabel('Amplitude', fontsize=20, fontweight='normal')
    plt.xlabel('Time, log scale', fontsize=20, fontweight='normal')
    plt.title(f'FFT spectrum of the signal "{signal_type}" in {year}', 
              fontsize=20, fontweight='normal', pad=20)

    # Styled grid (matches heatmap)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')
    
    # =========================================================================
    # VERTICAL REFERENCE LINES - Matches heatmap styling
    # =========================================================================
    
    # Grey dashed lines (matches heatmap month separators)
    for xc in xcoords:
        plt.axvline(x=xc,linewidth=1.2, color='black', linestyle='--', alpha = 0.5)
    plt.xticks(xcoords,ylabel)
    plt.show(block=False)
    
    return fig


def plot_EPN(emax, pmax, n, uf, serv, time_scales, satisfactions, scenario_name):
    """
    Plot Energy-Power-Number analysis results
    """
    # Aesthetic settings
    sns.set()
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")
    plt.rc('text', usetex=False)

    markers = ['o', 'v', 's', '^', 'o', 'v', 's', '^', 'o', 'v', 's', '^']
    markers = ''.join(markers)
    mark_size = 10

    xcoords = [24, 7 * 24, 30 * 24, 365 * 24]  # verticals black line to spot the day, the week and the month

    labels = [str(satis)+' %' for satis in satisfactions]

    # Figure settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.rc('lines', linewidth=3)

    fwidth = 12.  # total width of the figure in inches
    fheight = 8.  # total height of the figure in inches

    fig = plt.figure(figsize=(fwidth, fheight))

    # Define margins -> size in inches / figure dimension
    left_margin = 0.95 / fwidth
    right_margin = 0.2 / fwidth
    bottom_margin = 0.5 / fheight
    top_margin = 0.25 / fheight

    # Create axes
    x = left_margin  # horiz. position of bottom-left corner
    y = bottom_margin  # vert. position of bottom-left corner
    w = 1 - (left_margin + right_margin)  # width of axes
    h = 1 - (bottom_margin + top_margin)  # height of axes

    ax = fig.add_axes([x, y, w, h])

    # Define the Ylabel position
    xloc = 0.25 / fwidth
    yloc = y + h / 2.

    plt.close('all')
    
    # Usage factor
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()

    plt.ylabel(r"Utilization factor factor ($\%$)")
    plt.xscale('log')
    plt.xlabel("cycle length (h)")
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])
    plt.grid(True, which="both")
    lines = plt.plot(time_scales, uf)
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()
    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')
    plt.ylim(0, 105)

    plt.title(scenario_name)

    # Energy
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()
    plt.yscale('log')
    plt.ylabel('Energy (MWh)')
    plt.xlabel("cycle length (h)")
    plt.xscale('log')
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])

    plt.grid(True, which="both")
    lines = plt.plot(time_scales, emax)
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()
    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')

    plt.title(scenario_name)

    # Service
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()
    plt.xscale('log')
    plt.xlabel("cycle length (h)")
    plt.ylabel(r"E$\cdot n_{cycles}$ (MWh/year)")
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])
    lines = plt.plot(time_scales, serv)
    plt.grid(True, which="both")
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--', label='time')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()

    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    
    plt.title(scenario_name)

    plt.show()


    def plot_EPN_scenarios_plotly(Emax, UF, Serv, time_scales, scenario_names_list,
                               satisfactions=[95.],
                               title="Energy Scenario Comparison",
                               metrics='all',
                               show_plots=True,
                               save_html=None):
    """
    Plot EPN comparison for multiple energy scenarios using Plotly (interactive).
    Creates separate interactive figures for selected metrics.
    
    Parameters
    ----------
    Emax : list of arrays
        Energy results for each scenario
    UF : list of arrays
        Utilization factor results for each scenario
    Serv : list of arrays
        Service results for each scenario
    time_scales : list
        List of time scales (hours)
    scenario_names_list : list of str
        Names of each scenario
    satisfactions : list
        Satisfaction rates used (default: [95.])
    title : str
        Title prefix for figures
    metrics : str or list
        Which metrics to plot. Options:
        - 'all' : plot all metrics (UF, Energy, Service)
        - 'energy' : plot only Energy
        - 'uf' : plot only Utilization Factor
        - 'service' : plot only Service
        - ['energy', 'uf'] : plot specific metrics (list)
    show_plots : bool
        Whether to display plots
    save_html : str or None
        Directory to save HTML files (None = don't save)
    
    Returns
    -------
    figures : dict
        Dictionary of Plotly figure objects (only contains requested metrics)
    """
    
    # Satisfaction index
    sat_idx = 0
    
    # Colors for scenarios (colorblind-friendly palette)
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
        '#000000',  # Black
    ]
    
    # Markers
    markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x', 'star']
    
    # Reference lines positions and labels
    ref_lines = [
        (24, 'day'),
        (168, 'week'),
        (720, 'month'),
        (8760, 'year')
    ]
    
    # X-axis tick configuration
    tickvals = [0.75, 3, 10, 24, 168, 720, 8760]
    ticktext = ['0.75', '3', '10', 'day', 'week', 'month', 'year']
    
    figures = {}
    
    # =========================================================================
    # DETERMINE WHICH METRICS TO PLOT
    # =========================================================================
    
    if metrics == 'all':
        metrics_to_plot = ['uf', 'energy', 'service']
    elif isinstance(metrics, str):
        metrics_to_plot = [metrics.lower()]
    else:
        metrics_to_plot = [m.lower() for m in metrics]
    
    # =========================================================================
    # FIGURE 1: UTILIZATION FACTOR
    # =========================================================================
    
    if 'uf' in metrics_to_plot:
        fig_uf = go.Figure()
        
        for i, name in enumerate(scenario_names_list):
            # Handle 1D or 2D arrays
            uf_data = UF[i][:, sat_idx] if UF[i].ndim > 1 else UF[i]
            
            fig_uf.add_trace(go.Scatter(
                x=time_scales,
                y=uf_data,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(symbol=markers[i % len(markers)], size=10)
            ))
        
        # Add reference lines
        for xval, label in ref_lines:
            fig_uf.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_uf.update_layout(
            title=f"{title} - Utilization Factor ({satisfactions[sat_idx]}% satisfaction)",
            xaxis_title="Cycle length (h)",
            yaxis_title="Utilization Factor (%)",
            xaxis_type="log",
            xaxis=dict(tickvals=tickvals, ticktext=ticktext),
            yaxis=dict(range=[0, 105]),
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        figures['uf'] = fig_uf
    
    # =========================================================================
    # FIGURE 2: ENERGY
    # =========================================================================
    
    if 'energy' in metrics_to_plot:
        fig_energy = go.Figure()
        
        for i, name in enumerate(scenario_names_list):
            emax_data = Emax[i][:, sat_idx] if Emax[i].ndim > 1 else Emax[i]
            
            fig_energy.add_trace(go.Scatter(
                x=time_scales,
                y=emax_data,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(symbol=markers[i % len(markers)], size=10)
            ))
        
        for xval, label in ref_lines:
            fig_energy.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_energy.update_layout(
            title=f"{title} - Energy Capacity ({satisfactions[sat_idx]}% satisfaction)",
            xaxis_title="Cycle length (h)",
            yaxis_title="Energy (MWh)",
            xaxis_type="log",
            yaxis_type="log",
            xaxis=dict(tickvals=tickvals, ticktext=ticktext),
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        figures['energy'] = fig_energy
    
    # =========================================================================
    # FIGURE 3: SERVICE
    # =========================================================================
    
    if 'service' in metrics_to_plot:
        fig_service = go.Figure()
        
        for i, name in enumerate(scenario_names_list):
            serv_data = Serv[i][:, sat_idx] if Serv[i].ndim > 1 else Serv[i]
            
            fig_service.add_trace(go.Scatter(
                x=time_scales,
                y=serv_data,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(symbol=markers[i % len(markers)], size=10)
            ))
        
        for xval, label in ref_lines:
            fig_service.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_service.update_layout(
            title=f"{title} - Service ({satisfactions[sat_idx]}% satisfaction)",
            xaxis_title="Cycle length (h)",
            yaxis_title="E × n_cycles (MWh/year)",
            xaxis_type="log",
            xaxis=dict(tickvals=tickvals, ticktext=ticktext),
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        figures['service'] = fig_service
    
    # =========================================================================
    # SAVE HTML (optional)
    # =========================================================================
    
    if save_html:
        import os
        os.makedirs(save_html, exist_ok=True)
        
        for name, fig in figures.items():
            filepath = os.path.join(save_html, f"EPN_scenarios_{name}.html")
            fig.write_html(filepath)
            print(f"✅ Saved: {filepath}")
    
    # =========================================================================
    # SHOW PLOTS
    # =========================================================================
    
    if show_plots:
        for fig in figures.values():
            fig.show()
    
    return figures


def plot_EPN_scenarios_plotly_combined(Emax, UF, Serv, time_scales, scenario_names_list,
                                        satisfactions=[95.],
                                        title="Energy Scenario Comparison",
                                        height=500,
                                        show_plot=True,
                                        save_html=None):
    """
    Plot EPN comparison in a single Plotly figure with 3 subplots.
    
    Parameters
    ----------
    Same as plot_EPN_scenarios_plotly()
    height : int
        Figure height in pixels (default: 500)
    
    Returns
    -------
    fig : Plotly figure
    """
    
    sat_idx = 0
    
    # Colors and markers
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377', '#BBBBBB', '#000000']
    markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x', 'star']
    
    # Reference lines
    ref_lines = [(24, 'day'), (168, 'week'), (720, 'month'), (8760, 'year')]
    tickvals = [0.75, 3, 10, 24, 168, 720, 8760]
    ticktext = ['0.75', '3', '10', 'day', 'week', 'month', 'year']
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Utilization Factor', 'Energy Capacity', 'Service'),
        horizontal_spacing=0.08
    )
    
    # Add traces for each scenario
    for i, name in enumerate(scenario_names_list):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # UF data
        uf_data = UF[i][:, sat_idx] if UF[i].ndim > 1 else UF[i]
        fig.add_trace(go.Scatter(
            x=time_scales, y=uf_data,
            mode='lines+markers', name=name,
            line=dict(color=color, width=2),
            marker=dict(symbol=marker, size=8),
            legendgroup=name,
            showlegend=True
        ), row=1, col=1)
        
        # Energy data
        emax_data = Emax[i][:, sat_idx] if Emax[i].ndim > 1 else Emax[i]
        fig.add_trace(go.Scatter(
            x=time_scales, y=emax_data,
            mode='lines+markers', name=name,
            line=dict(color=color, width=2),
            marker=dict(symbol=marker, size=8),
            legendgroup=name,
            showlegend=False
        ), row=1, col=2)
        
        # Service data
        serv_data = Serv[i][:, sat_idx] if Serv[i].ndim > 1 else Serv[i]
        fig.add_trace(go.Scatter(
            x=time_scales, y=serv_data,
            mode='lines+markers', name=name,
            line=dict(color=color, width=2),
            marker=dict(symbol=marker, size=8),
            legendgroup=name,
            showlegend=False
        ), row=1, col=3)
    
    # Add reference lines to all subplots
    for col in [1, 2, 3]:
        for xval, label in ref_lines:
            fig.add_vline(x=xval, line_dash="dash", line_color="gray", opacity=0.3, row=1, col=col)
    
    # Update axes
    # Subplot 1: UF
    fig.update_xaxes(type="log", tickvals=tickvals, ticktext=ticktext, title_text="Cycle length (h)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization Factor (%)", range=[0, 105], row=1, col=1)
    
    # Subplot 2: Energy
    fig.update_xaxes(type="log", tickvals=tickvals, ticktext=ticktext, title_text="Cycle length (h)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Energy (MWh)", row=1, col=2)
    
    # Subplot 3: Service
    fig.update_xaxes(type="log", tickvals=tickvals, ticktext=ticktext, title_text="Cycle length (h)", row=1, col=3)
    fig.update_yaxes(title_text="E × n_cycles (MWh/year)", row=1, col=3)
    
    # Update layout
    fig.update_layout(
        title=f"{title} ({satisfactions[sat_idx]}% satisfaction)",
        height=height,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    # Save HTML
    if save_html:
        fig.write_html(save_html)
        print(f"✅ Saved: {save_html}")
    
    # Show
    if show_plot:
        fig.show()
    
    return fig