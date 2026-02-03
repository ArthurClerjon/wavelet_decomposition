# 🚀 Quick Start Guide - Streamlit Wavelet Decomposition App

## 📋 Table of Contents
1. [Installation](#installation)
2. [Running the App](#running-the-app)
3. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
4. [Common Issues](#common-issues)

---

## 📥 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- streamlit (web interface)
- numpy, pandas (data processing)
- matplotlib, seaborn (visualization)
- scipy (scientific computing)
- openpyxl (Excel file reading)

---

## 🎯 Running the App

### Launch Command

```bash
streamlit run streamlit_app.py
```

**What happens:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

The app will automatically open in your default web browser!

---

## 📖 Step-by-Step Walkthrough

### **Step 1: Upload Your Data File** 📁

1. Click "Browse files" button
2. Select your Excel file (e.g., `input_time_series.xlsx`)
3. Configure import parameters:
   - **Original data points per day:** 48 (if your data is every 30 min)
   - **Interpolated points per day:** 64 (recommended)
   - **Days per year:** 365

4. Click "🔄 Import Data"

**Expected Output:**
```
✅ Data imported successfully!

Data Information:
- Available signals: Consumption, Wind, PV
- Years available: 2012, 2013, 2014, 2015, 2016, 2017, 2018 (7 years)
- Original sampling: 48 points/day
- Interpolated sampling: 64 points/day
- Days per year: 365
- Total points per year: 23,360
```

---

### **Step 2: Select Signal and Year** 🎯

1. **Signal type:** Select from dropdown
   - PV (solar photovoltaic)
   - Wind
   - Consumption

2. **Year:** Select which year to analyze
   - Example: 2012

3. **Country/Region:** Enter name
   - Example: France
   - Used for file organization

**Example Selection:**
```
Signal type: PV
Year: 2012
Country: France
```

---

### **Step 3: Configure Decomposition** ⚙️

Use the sliders to set decomposition levels:

```
Yearly wavelets:  ▓▓▓▓▓▓░░░░ 6
Weekly wavelets:  ▓▓▓░░░░░░░ 3
Daily wavelets:   ▓▓▓▓▓▓░░░░ 6
```

**What these mean:**
- **Yearly:** How finely to divide seasonal patterns
- **Weekly:** How finely to divide weekly patterns
- **Daily:** How finely to divide daily patterns

**Recommendation:** Keep defaults (6, 3, 6) for first analysis

---

### **Step 4: Run Decomposition** 🚀

1. Leave "Recompute translations" **unchecked** (faster)
2. Click "🚀 Run Wavelet Decomposition"
3. Wait for completion (typically 10-60 seconds)

**Expected Output:**
```
✅ Decomposition complete!
- Translation file: results/France/translations/trans_France_PV.pkl
- Matrix file: results/France/matrices/A_2012.npz
- Betas computed for 2012
```

---

### **Step 5: Generate Visualizations** 📈

#### **Option A: Heatmap Only**

1. Check ✅ "📊 Plot Heatmap"
2. Configure color scale:
   ```
   Minimum:  -0.1
   Maximum:   0.1
   Center:    0.0
   ```

3. Select time scales to display:
   ```
   ✅ 24.0h   (daily)
   ✅ 168.0h  (weekly)
   ✅ 8760.0h (yearly)
   ```

4. Click "📊 Generate Visualizations"

**Result:** Heatmap showing wavelet coefficients across time and selected scales

#### **Option B: Heatmap + FFT**

1. Check ✅ "📊 Plot Heatmap"
2. Check ✅ "📉 Plot FFT Spectrum"
3. Click "📊 Generate Visualizations"

**Result:** Both plots displayed for comparison

---

### **Step 6: Reconstruct Signal** 🔄

#### **Example: Isolate Daily Pattern**

1. Scroll to "Step 6: Signal Reconstruction"
2. In "Sub-daily scales" section, check:
   ```
   ✅ 24h (day)
   ```
3. Uncheck all other scales
4. Leave "Add offset" **unchecked**
5. Click "🔄 Reconstruct Signal"

**Result:** Signal showing only the daily cycle component

#### **Example: Seasonal Pattern Only**

1. In "Seasonal scales" section, check:
   ```
   ✅ 8760h (year)
   ```
2. Uncheck all other scales
3. Click "🔄 Reconstruct Signal"

**Result:** Signal showing only the seasonal/yearly variation

#### **Example: Weekly + Seasonal**

1. Select:
   ```
   ✅ 168h (week)
   ✅ 8760h (year)
   ```
2. Click "🔄 Reconstruct Signal"

**Result:** Combined weekly and seasonal patterns

---

## 💡 Complete Example Workflow

### **Scenario: Analyze PV Solar Production Patterns**

```python
# What to do:
1. Upload: input_time_series.xlsx
2. Select: Signal = PV, Year = 2012, Country = France
3. Decomposition: Keep defaults (6, 3, 6)
4. Run decomposition ✓
5. Visualize: Both heatmap and FFT ✓
6. Reconstruct with:
   - First: 24h only (see daily pattern)
   - Second: 168h only (see weekly pattern)
   - Third: 8760h only (see seasonal pattern)
```

**What you'll discover:**
- **Daily (24h):** Day/night production cycle
- **Weekly (168h):** Weekend vs weekday variations (if any)
- **Yearly (8760h):** Summer (high) vs winter (low) production

---

## 🎨 Interface Layout

```
┌─────────────────────────────────────────────────────┐
│  📊 Wavelet Decomposition Analysis                  │
│  [Interactive Streamlit Interface]                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📁 Step 1: Upload Data File                        │
│  ┌──────────────────────────────────────┐          │
│  │ [Browse files...] 📎                 │          │
│  │ Data points/day: [48] [64] [365]     │          │
│  │ [🔄 Import Data]                     │          │
│  └──────────────────────────────────────┘          │
│                                                      │
│  🎯 Step 2: Select Signal and Year                  │
│  ┌──────────────────────────────────────┐          │
│  │ Signal: [PV ▼]  Year: [2012 ▼]      │          │
│  │ Country: [France]                    │          │
│  └──────────────────────────────────────┘          │
│                                                      │
│  ⚙️ Step 3: Decomposition Parameters                │
│  ┌──────────────────────────────────────┐          │
│  │ Yearly:  ▓▓▓▓▓▓░░░░ 6               │          │
│  │ Weekly:  ▓▓▓░░░░░░░ 3               │          │
│  │ Daily:   ▓▓▓▓▓▓░░░░ 6               │          │
│  └──────────────────────────────────────┘          │
│                                                      │
│  🚀 Step 4: Run Decomposition                       │
│  ┌──────────────────────────────────────┐          │
│  │ [🚀 Run Wavelet Decomposition]       │          │
│  └──────────────────────────────────────┘          │
│                                                      │
│  📈 Step 5: Visualization                           │
│  ┌──────────────────────────────────────┐          │
│  │ ☑ Plot Heatmap  ☐ Plot FFT          │          │
│  │ [📊 Generate Visualizations]         │          │
│  └──────────────────────────────────────┘          │
│                                                      │
│  🔄 Step 6: Signal Reconstruction                   │
│  ┌──────────────────────────────────────┐          │
│  │ Time scales: ☑ 24h ☑ 168h ☑ 8760h   │          │
│  │ [🔄 Reconstruct Signal]              │          │
│  └──────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

---

## ❌ Common Issues & Solutions

### **Issue 1: "No module named 'streamlit'"**

**Solution:**
```bash
pip install streamlit
```

### **Issue 2: "File not found" error after decomposition**

**Cause:** FileManager couldn't find the matrix file

**Solution:**
1. Check that decomposition completed successfully
2. Look for the success message with file paths
3. Verify `results/` directory exists
4. Try running decomposition again

### **Issue 3: Import error "Cannot read Excel file"**

**Solution:**
```bash
pip install openpyxl
```

### **Issue 4: Plots not displaying**

**Cause:** Matplotlib backend issue

**Solution:**
1. Restart the Streamlit app
2. Check browser console for errors
3. Try a different browser

### **Issue 5: "Session state key not found"**

**Cause:** Steps executed out of order

**Solution:**
1. Start from Step 1 (Upload)
2. Complete each step in sequence
3. Don't skip the decomposition step

---

## 🔧 Configuration Tips

### **For Large Files:**

If your Excel file is very large (>100MB):
```python
# In streamlit_app.py, add:
st.set_option('server.maxUploadSize', 500)  # 500 MB
```

### **For Multiple Users:**

Run on a server:
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### **For Development:**

Enable auto-reload:
```bash
streamlit run streamlit_app.py --server.runOnSave true
```

---

## 📊 Expected Processing Times

| Operation | Typical Duration | Notes |
|-----------|-----------------|-------|
| Data Import | 5-10 seconds | Depends on file size |
| Decomposition (1 year) | 15-60 seconds | Depends on parameters |
| Heatmap generation | 2-5 seconds | Fast |
| FFT spectrum | 1-2 seconds | Very fast |
| Reconstruction | 3-10 seconds | Depends on # scales |

**Total workflow:** 2-5 minutes for complete analysis of one year

---

## ✅ Success Indicators

You know everything is working when you see:

1. **After Import:**
   ```
   ✅ Data imported successfully!
   Data Information: [shows years, signals, etc.]
   ```

2. **After Decomposition:**
   ```
   ✅ Decomposition complete!
   - Translation file: [path]
   - Matrix file: [path]
   ```

3. **After Visualization:**
   ```
   ✅ Heatmap generated!
   ✅ FFT spectrum generated!
   ```

4. **After Reconstruction:**
   ```
   ✅ Signal reconstructed with X time scales!
   [Shows statistics: Mean, Std Dev, Min, Max]
   ```

---

## 🎓 Next Steps

Once you're comfortable with the basics:

1. **Try different signals:** Compare Wind vs PV vs Consumption
2. **Try different years:** See how patterns change over time
3. **Experiment with parameters:** Adjust vy, vw, vd to see effects
4. **Combine time scales:** Mix daily + seasonal for interesting patterns
5. **Compare reconstructions:** Original vs reconstructed signal

---

## 🆘 Getting Help

If stuck:
1. ✅ Check this guide
2. ✅ Read error messages carefully
3. ✅ Verify data format
4. ✅ Try with example data first
5. ✅ Restart the app

---

## 🎉 You're Ready!

**Command to start:**
```bash
streamlit run streamlit_app.py
```

**Then:** Follow the 6 steps in the interface!

Happy analyzing! 📊✨
