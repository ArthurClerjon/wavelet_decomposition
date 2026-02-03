# Wavelet Decomposition Interface Documentation

## Objective

The Wavelet Decomposition Interface is designed to analyze time series data, particularly focusing on energy consumption, wind power, and solar PV production. The interface provides tools for wavelet decomposition, reconstruction, and visualization of the results. It also allows users to analyze the flexibility requirements of the energy system based on the wavelet decomposition.

## Features

1. **Time Series Selection**: Users can select a time series from existing Excel files.
2. **Time Series Conversion**: The selected time series is converted to the appropriate format with ndpd=64.
3. **Wavelet Decomposition**: The interface performs wavelet decomposition on the converted time series.
4. **Reconstruction and Plotting**: Users can reconstruct the time series for given time scales and plot the heatmap for a given time scale. Additionally, users can plot the FFT of the signal for comparison purposes.
5. **Energy Mix Selection and Analysis**: Users can select an energy mix (intermittent supply and demand) and carry out the wavelet decomposition on the residual demand (load minus intermittent supply). From the decomposition, the interface performs the EPN analysis to analyze the flexibility requirements.
6. **Saving Results**: The interface saves the matrix (A), translations (trans), and wavelet decomposition results (betas) in a tidy repository for enhanced time calculation.
7. **Streamlit Interface**: The interface is built using Streamlit to provide a user-friendly interface for the wavelet decomposition and analysis process.

## Usage

1. **Selecting a Time Series**: Use the interface to select a time series from the available Excel files.
2. **Converting the Time Series**: The interface will automatically convert the selected time series to the appropriate format.
3. **Performing Wavelet Decomposition**: The interface will perform the wavelet decomposition on the converted time series.
4. **Reconstructing and Plotting**: Use the interface to reconstruct the time series for given time scales and plot the heatmap for a given time scale. You can also plot the FFT of the signal for comparison purposes.
5. **Selecting an Energy Mix and Analyzing Flexibility Requirements**: Use the interface to select an energy mix and perform the wavelet decomposition on the residual demand. The interface will then perform the EPN analysis to analyze the flexibility requirements.
6. **Saving Results**: The interface will save the matrix (A), translations (trans), and wavelet decomposition results (betas) in a tidy repository.
7. **Using the Streamlit Interface**: Use the Streamlit interface to interact with the wavelet decomposition and analysis process.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- XlsxWriter
- Pickle

## Installation

1. Clone the repository: `git clone https://github.com/ArthurClerjon/wavelet_decomposition.git`
2. Navigate to the project directory: `cd wavelet_decomposition`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the Streamlit interface: `streamlit run interface_prototype.py`

## Conclusion

The Wavelet Decomposition Interface provides a comprehensive tool for analyzing time series data and understanding the flexibility requirements of the energy system. With its user-friendly interface and powerful features, it is a valuable resource for researchers and analysts in the field of energy systems.
