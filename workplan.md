# Wavelet Decomposition Interface Workplan

## Objective

The objective of this workplan is to detail the required steps and functions to be implemented in the Wavelet Decomposition Interface to achieve the desired functionality.

## Workplan

1. **Time Series Selection Function**
   - **Function**: `import_excel`
   - **Input**: Path to the Excel file containing the time series data.
   - **Output**: Selected time series data.
   - **Description**: This function will allow users to select a time series from existing Excel files.

2. **Time Series Conversion Function**
   - **Function**: `import_excel` with interpolation parameter set to True.
   - **Input**: Selected time series data.
   - **Output**: Converted time series data with ndpd=64.
   - **Description**: This function will convert the selected time series to the appropriate format with ndpd=64.
   - At this point, this function also plot the time series. 
   Todo : Once the prototype work, create another function for ploting the time series.

3. **Wavelet Decomposition Function**
   - **Function**: `wavelet_decomposition_single_TS`
   - **Input**: Converted time series data AND the parameters of those data, inherited from import_excel( functgion
   - **Output**: Wavelet decomposition results with 3 files : the translations calculated for the time serie (TS) selected, the wavelet_matric wich has been calculated. Those data are stored once for all in a precis directory to make sure we don't need to recompute them. Eventually, the results of the wavelet decomposition (betas coefficients) are also computed
   - **Description**: This function will perform wavelet decomposition on the converted time series.

4. **Reconstruction and Plotting Functions**
   - **Function**: `reconstruct` and `plot_betas_heatmap`
   - **Input**: Wavelet decomposition results.
   - **Output**: Reconstructed time series and heatmap plot.
   - **Description**: These functions will allow users to reconstruct the time series for given time scales and plot the heatmap for a given time scale.

5. **FFT Plotting Function**
   - **Function**: `fft`
   - **Input**: Converted time series data.
   - **Output**: FFT plot.
   - **Description**: This function will plot the FFT of the signal for comparison purposes.

6. **Energy Mix Selection and Analysis Functions**
   - **Function**: `wavelet_decomposition_single_TS` and `calc_epn`
   - **Input**: Selected energy mix (intermittent supply and demand) and residual demand (load minus intermittent supply).
   - **Output**: Wavelet decomposition results and EPN analysis results.
   - **Description**: These functions will allow users to select an energy mix and perform the wavelet decomposition on the residual demand. The EPN analysis will then be performed to analyze the flexibility requirements.

7. **Saving Results Functions**
   - **Function**: `create_directory` and `pkl.dump`
   - **Input**: Matrix (A), translations (trans), and wavelet decomposition results (betas).
   - **Output**: Saved matrix (A), translations (trans), and wavelet decomposition results (betas) in a tidy repository.
   - **Description**: These functions will save the matrix (A), translations (trans), and wavelet decomposition results (betas) in a tidy repository for enhanced time calculation.

8. **Streamlit Interface Creation**
   - **Function**: Streamlit functions and components.
   - **Input**: Wavelet decomposition results and EPN analysis results.
   - **Output**: User-friendly Streamlit interface.
   - **Description**: This step will involve creating a user-friendly Streamlit interface to interact with the wavelet decomposition and analysis process.

## Conclusion

This workplan outlines the required steps and functions to be implemented in the Wavelet Decomposition Interface to achieve the desired functionality. By following this workplan, we can ensure that the interface is developed efficiently and effectively.
