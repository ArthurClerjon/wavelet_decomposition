# 📊 Wavelet Decomposition Analysis - Streamlit Interface

Interactive web application for analyzing time series using wavelet decomposition, based on the Clerjon & Perdu (2019) methodology.

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Run the Application**

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your web browser at `http://localhost:8501`

---

## 📋 Features

### ✅ **Complete Workflow**

1. **📁 File Upload**
   - Upload Excel files with time series data
   - Supports Consumption, Wind, and PV signals
   - Automatic data interpolation

2. **🎯 Signal Selection**
   - Choose signal type (PV, Wind, Consumption)
   - Select year to analyze
   - Configure country/region name

3. **⚙️ Decomposition Configuration**
   - Adjust yearly, weekly, and daily wavelet levels
   - 15 time scales from 0.75h to 8760h (1 year)
   - Option to recompute translations

4. **📈 Visualization**
   - **Heatmap**: Wavelet coefficients across time and scales
   - **FFT Spectrum**: Frequency domain analysis
   - Customizable color scales and time scale selection

5. **🔄 Signal Reconstruction**
   - Select specific time scales to include
   - Filter signal by frequency components
   - Optional DC offset inclusion

---

## 📁 File Structure

```
project/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt            # Python dependencies
├── file_manager.py            # File organization utilities
├── wavelet_decomposition.py   # Core decomposition functions
├── plots.py                   # Plotting functions
├── import_excel.py            # Excel import utilities
└── config.py                  # Configuration settings
```

---

## 📊 Data Format

### Excel File Requirements

Your Excel file should have the following structure:

| Date | Consumption | Wind | PV |
|------|-------------|------|----|
| 2012-01-01 00:00 | 54231.5 | 1234.2 | 0.0 |
| 2012-01-01 00:30 | 53987.3 | 1298.7 | 0.0 |
| ... | ... | ... | ... |

**Requirements:**
- ✅ Columns: `Consumption`, `Wind`, `PV` (at least one required)
- ✅ Consistent time intervals (e.g., 30-minute, 1-hour)
- ✅ Multiple years can be stacked vertically
- ✅ Data should be normalized to 1 MW per unit

---

## 🎨 Interface Sections

### **1. File Upload & Import**

```
📁 Step 1: Upload Data File
├── Upload Excel file
├── Set data points per day (original)
├── Set interpolation target (64 points/day recommended)
└── Import button
```

**Output:** Data summary showing:
- Available signals
- Years in dataset
- Sampling rates
- Total data points

### **2. Signal & Year Selection**

```
🎯 Step 2: Select Signal and Year
├── Signal type dropdown (PV, Wind, Consumption)
├── Year selector
└── Country/Region name
```

### **3. Decomposition Parameters**

```
⚙️ Step 3: Decomposition Parameters
├── Yearly wavelets slider (1-10, default: 6)
├── Weekly wavelets slider (1-10, default: 3)
├── Daily wavelets slider (1-10, default: 6)
└── Recompute translations checkbox
```

**Time Scales:**
- Sub-daily: 0.75h, 1.5h, 3h, 6h, 12h, 24h
- Weekly: 42h, 84h, 168h (week)
- Monthly: 273.75h, 547.5h, 1095h
- Seasonal: 2190h, 4380h (6 months), 8760h (year)

### **4. Visualization Options**

```
📈 Step 5: Visualization
├── Heatmap checkbox
│   ├── Color scale min/max
│   ├── Color scale center
│   └── Time scales to display
└── FFT Spectrum checkbox
```

**Heatmap Features:**
- Monthly x-axis with vertical separators
- Time scale labels (day, week, year)
- Diverging colormap (coolwarm)
- Modern sans-serif fonts

**FFT Spectrum Features:**
- Logarithmic x-axis
- Vertical reference lines (year, month, week, day, 12h, hour)
- Frequency domain analysis

### **5. Signal Reconstruction**

```
🔄 Step 6: Signal Reconstruction
├── Time scale checkboxes (grouped)
│   ├── Sub-daily scales
│   ├── Weekly-monthly scales
│   └── Seasonal scales
├── Add offset checkbox
└── Reconstruct button
```

**Output:**
- Reconstructed signal plot
- Statistics (mean, std, min, max)
- Download option (coming soon)

---

## 🎯 Usage Examples

### **Example 1: Analyze Daily Patterns**

1. Upload data file
2. Select signal: **PV**
3. Select year: **2012**
4. Run decomposition
5. Visualize: Check **Heatmap**, select **24h** time scale only
6. Reconstruct: Select only **24h** scale

→ **Result:** Daily cycle pattern isolated

### **Example 2: Compare Seasonal vs Weekly Patterns**

1. Upload data file
2. Select signal: **Consumption**
3. Run decomposition with default parameters
4. Visualize: **Heatmap** with **168h** and **8760h** scales
5. Reconstruct twice:
   - First with **168h** only (weekly pattern)
   - Then with **8760h** only (yearly/seasonal pattern)

→ **Result:** Compare weekly vs seasonal variations

### **Example 3: Full Spectrum Analysis**

1. Upload data file
2. Run decomposition
3. Check both **Heatmap** and **FFT Spectrum**
4. Compare:
   - Heatmap shows time-localized patterns
   - FFT shows overall frequency content

→ **Result:** Complete time-frequency analysis

---

## 📊 Decomposition Parameters Guide

### **Yearly Wavelets (vy)**
- **Low (1-3):** Coarse seasonal resolution
- **Medium (4-6):** Good seasonal detail (recommended)
- **High (7-10):** Very fine seasonal patterns (slower)

### **Weekly Wavelets (vw)**
- **Low (1-2):** Basic week/weekend pattern
- **Medium (3-4):** Good weekly detail (recommended)
- **High (5+):** Very detailed weekly patterns

### **Daily Wavelets (vd)**
- **Low (1-3):** Basic day/night pattern
- **Medium (4-6):** Good sub-daily detail (recommended)
- **High (7-10):** Very fine intra-day patterns (slower)

**💡 Tip:** Start with defaults (vy=6, vw=3, vd=6) and adjust based on results.

---

## 🔧 Troubleshooting

### **Issue: "Matrix file not found"**

**Solution:** Make sure decomposition completed successfully. Check that the `results/` directory was created.

### **Issue: "Data import failed"**

**Solution:** 
- Verify Excel file has correct column names
- Check for missing data or NaN values
- Ensure consistent time intervals

### **Issue: "Reconstruction error"**

**Solution:**
- Select at least one time scale
- Make sure decomposition was run first
- Check that the year matches

### **Issue: "Memory error"**

**Solution:**
- Reduce number of years analyzed (process one at a time)
- Lower decomposition parameters (vy, vw, vd)
- Close other applications

---

## ⚡ Performance Tips

1. **Start small:** Process one year at a time initially
2. **Default parameters:** Use vy=6, vw=3, vd=6 for balanced performance
3. **Cache translations:** Don't recompute unless necessary
4. **Selective visualization:** Choose specific time scales instead of all 15

---

## 📚 Methodology Reference

This application implements the wavelet decomposition methodology described in:

**Clerjon, A., & Perdu, F. (2019).** "Wavelet-based sizing of decarbonized energy systems"  
*Applied Energy*

**Key features:**
- Square wavelet basis functions
- Optimal translation computation
- Multi-scale decomposition
- Time-frequency analysis

---

## 🛠️ Advanced Configuration

### **Custom Time Scales**

Edit `time_scales` in the code to use custom scales:

```python
time_scales = [1., 2., 4., 8., 24., 168., 720., 8760.]  # Example
```

### **File Organization**

The app creates this structure:

```
results/
└── {country}/
    ├── translations/
    │   └── trans_{country}_{signal}.pkl
    ├── matrices/
    │   └── A_{year}.npz
    └── betas/
        └── betas_{country}_{signal}_{year}.pkl
```

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Verify data format matches requirements

---

## 🎓 Learning Resources

**Understanding Wavelets:**
- Start with daily (24h) and yearly (8760h) scales
- Observe how patterns repeat
- Compare with FFT spectrum

**Best Practices:**
1. Always visualize both heatmap and FFT
2. Start with few time scales, add more as needed
3. Compare reconstructed signals with originals
4. Document parameter choices

---

## ✅ Quick Checklist

Before starting:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Excel file prepared with correct format
- [ ] At least 1 year of data available

During analysis:
- [ ] Data imported successfully
- [ ] Signal and year selected
- [ ] Decomposition parameters configured
- [ ] Decomposition completed without errors
- [ ] Visualizations generated
- [ ] Reconstruction verified

---

## 🎉 Happy Analyzing!

This interface makes wavelet decomposition accessible and interactive. 
Experiment with different settings to gain insights into your time series data!

**Version:** 1.0  
**Last Updated:** January 2026
