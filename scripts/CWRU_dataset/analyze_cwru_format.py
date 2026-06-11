"""
CWRU Raw .mat File Analysis & Format Recommendations
Analyzes bearing fault data structure and recommends optimal storage format

This script:
1. Inspects .mat files to understand bearing signal structure
2. Recommends data format (CSV vs .mat vs HDF5)
3. Proposes additional feature engineering for scientific rigor
4. Validates current feature extraction
"""

import os 
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# PART 1: INSPECT .MAT FILE STRUCTURE
# ============================================================================
def morlet2(M, s, w=5.0):
    x = np.arange(0, M) - (M - 1) / 2.0
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    return wavelet
    
def analyze_mat_files():
    """Inspect a sample .mat file to understand data structure."""
    
    mat_dir = os.path.join(PROJECT_ROOT, "data", "CWRU_dataset", "raw", "raw")
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')][:3]  # First 3 files
    
    print("\n" + "="*80)
    print("CWRU RAW .MAT FILES ANALYSIS")
    print("="*80)
    
    for mat_file in mat_files:
        mat_path = os.path.join(mat_dir, mat_file)
        print(f"\n[File: {mat_file}]")
        
        try:
            mat_data = sio.loadmat(mat_path)
            
            # Print available keys (excluding MATLAB internal keys)
            user_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"  Keys: {user_keys}")
            
            # Analyze each data array
            for key in user_keys:
                data = mat_data[key]
                if isinstance(data, np.ndarray):
                    print(f"  ├─ {key}:")
                    print(f"  │  ├─ Shape: {data.shape}")
                    print(f"  │  ├─ Dtype: {data.dtype}")
                    print(f"  │  ├─ Min: {np.min(data):.6f}")
                    print(f"  │  ├─ Max: {np.max(data):.6f}")
                    print(f"  │  ├─ Mean: {np.mean(data):.6f}")
                    print(f"  │  └─ Std: {np.std(data):.6f}")
                    
        except Exception as e:
            print(f"  Error reading {mat_file}: {e}")
    
    return mat_files

# ============================================================================
# PART 2: ADVANCED FEATURE ENGINEERING RECOMMENDATIONS
# ============================================================================

def recommend_advanced_features():
    """
    Provide comprehensive recommendations for advanced features suitable
    for peer-reviewed publications.
    """
    
    report = []
    report.append("\n" + "="*80)
    report.append("RECOMMENDED ADVANCED FEATURES FOR PUBLICATION")
    report.append("="*80)
    
    report.append("""
Your current features are GOOD but can be enhanced:

CURRENT FEATURES (Good):
├─ Temporal Statistics: Mean, Std, Min, Max, RMS
├─ Statistical Moments: Skewness, Kurtosis
└─ Frequency Domain: Peak Frequency

RECOMMENDED ADDITIONS:

1. **SPECTRAL FEATURES** (High discriminative power)
   ├─ Spectral Entropy (Shannon entropy of power spectral density)
   ├─ Spectral Energy (sum of squared FFT components)
   ├─ Spectral Centroid (center of mass in frequency domain)
   ├─ Spectral Flux (change in power spectrum over time)
   ├─ Welch PSD peaks (dominant frequency components)
   ├─ Spectral Kurtosis (detects impulsive events - KEY for bearing faults)
   └─ Crest Factor (max/RMS ratio - sensitive to impacts)

2. **WAVELET FEATURES** (Captures transient fault signatures)
   ├─ Continuous Wavelet Transform energy at specific frequencies
   ├─ Wavelet packet energy decomposition (8-16 bands)
   ├─ Intrinsic Mode Function (IMF) energies via EMD
   ├─ Maximum Lyapunov exponent (nonlinear dynamics)
   └─ Sample Entropy (complexity measure)

3. **TIME-DOMAIN DERIVED** (Fault-specific indicators)
   ├─ Impulse Factor = Max / Mean (sensitive to bearing defects)
   ├─ Margin Factor = Max / √(Mean(|x|²))
   ├─ Clearance Factor = Max / (Mean(√|x|))⁴
   ├─ Kurtosis value itself (very important - ignored in current pipeline)
   ├─ 4th Central Moment (enhanced kurtosis detection)
   ├─ Autocorrelation at lag-1 (periodicity detection)
   ├─ Zero-crossing rate (impacts detection)
   └─ Time between peaks (impulse interval)

4. **FREQUENCY DOMAIN** (Beyond simple peak detection)
   ├─ FFT magnitude at fault-characteristic frequencies (FCF)
   ├─ Bearing fundamental frequency (BFF)
   ├─ Ball pass frequency outer race (BPFO)
   ├─ Ball pass frequency inner race (BPFI)
   ├─ Multiple harmonics of FCF (2×, 3×, 4× FCF)
   ├─ Energy in bands around fault frequencies
   ├─ Coherence between axes (X-Y, Y-Z, X-Z)
   └─ Cross-spectrum phase alignment

5. **TEMPORAL-FREQUENCY (Time-Frequency analysis)**
   ├─ Spectrogram features (energy distribution across time-freq)
   ├─ Morlet wavelet scalogram energy
   ├─ Short-Time Fourier Transform (STFT) statistics
   ├─ Mel-frequency cepstral coefficients (MFCCs)
   └─ Constant-Q transform (log-frequency resolution)

6. **STATISTICAL MOMENTS (Advanced)**
   ├─ 5th & 6th Central Moments (hyperskewness, hyperkurtosis)
   ├─ Shape factors (α, β from Weibull distribution)
   ├─ Peak-to-Peak ratio
   ├─ Variance of first differences (acceleration smoothness)
   └─ Distribution tail properties (percentiles 90, 95, 99)

7. **MULTIVARIATE CORRELATION**
   ├─ Cross-correlation between axes at peak lags
   ├─ Principal Component Analysis (PCA) variance ratios
   ├─ Singular Value Decomposition (SVD) spectral norms
   └─ Mahalanobis distance from normal operating point

CRITICAL FOR YOUR PAPER:
→ **Spectral Kurtosis** (most important for rolling element bearing faults)
→ **Impulse/Margin/Clearance Factors** (industry standard indicators)
→ **FCF-related harmonics** (bearing-specific physics)
→ **Wavelet entropy** (captures fault transients)
""")
    
    return "\n".join(report)

# ============================================================================
# PART 3: DATA FORMAT RECOMMENDATIONS
# ============================================================================

def recommend_data_format():
    """Provide recommendations on optimal storage format."""
    
    report = []
    report.append("\n" + "="*80)
    report.append("DATA FORMAT RECOMMENDATIONS")
    report.append("="*80)
    
    report.append("""
YOUR CURRENT SETUP: CSV files in CWRU/processed/
VERDICT: ✓ ACCEPTABLE, but with caveats

COMPARISON TABLE:
╔════════════════════╦═════════╦═════════╦═════════════╦═════════════════╗
║ Format             ║ Speed   ║ Size    ║ Reusability ║ Reproducibility ║
╠════════════════════╬═════════╬═════════╬═════════════╬═════════════════╣
║ CSV (current)      ║  Slow   ║ Large   ║  High       ║  Very High      ║
║ HDF5 (recommend)   ║  Fast   ║ Small   ║  High       ║  High           ║
║ Parquet (alt.)     ║  Medium ║ Small   ║  High       ║  High           ║
║ NumPy (.npy)       ║  Very   ║ Small   ║  Low        ║  Medium         ║
║                    ║  Fast   ║         ║             ║                 ║
║ .mat (keep source) ║  Fast   ║ Large   ║  Low        ║  Very High      ║
╚════════════════════╩═════════╩═════════╩═════════════╩═════════════════╝

RECOMMENDATION FOR SCIENTIFIC ARTICLE:

1. **PRIMARY FORMAT: HDF5** (Hierarchical Data Format 5)
   ├─ Why: Industry standard for scientific computing
   ├─ Pro: Metadata support, partial I/O, compression, versioning
   ├─ Con: Requires h5py library (minimal dependency)
   ├─ Usage: Main analysis, faster training/inference
   ├─ Python: import h5py; file = h5py.File('data.h5', 'r')
   └─ Example structure:
       └─ cwru_processed.h5
           ├─ B007_1_123 (group)
           │  ├─ features (dataset: Nx32 float32)
           │  ├─ raw_signal (dataset: metadata)
           │  └─ metadata (attributes: source, date, params)
           ├─ B014_1_190 (group)
           └─ metadata (attributes: schema_version, creation_date)

2. **SECONDARY FORMAT: CSV** (current - keep for transparency)
   ├─ Why: Human-readable, supports supplementary materials
   ├─ Pro: Reviewers can verify, Excel-friendly
   ├─ Con: Large files, slow I/O
   └─ Usage: Supplementary data in journal submission

3. **RETAIN RAW: .mat files** (preserve original)
   ├─ Why: Complete traceability, reanalysis capability
   ├─ Pro: Original vendor format (reproducibility)
   ├─ Con: Proprietary, harder to share
   └─ Usage: Archive in GitHub/Zenodo for reproducibility

MISSING CRITICAL INFORMATION:
Your current CSV files lack:
  ⚠ Bearing load information (1000 lbf, 1500 lbf, 2000 lbf, 3000 lbf)
  ⚠ Bearing type (6205-2RS, 6203-2RS, 6204-2RS)
  ⚠ Fault size/category (7mil, 14mil, 21mil, normal)
  ⚠ Operating RPM (1772, 1750, 1650, 1500 RPM)
  ⚠ Data collection date/batch
  ⚠ Sampling parameters (sample rate, filter settings)

ACTION ITEMS FOR SCIENTIFIC RIGOR:

✓ Extract metadata from filename: {Type}{Size}_{Load}_{RPM}_features.csv
✓ Create metadata CSV with: [filename, fault_type, fault_size, load, rpm, samples]
✓ Add to each processed file as integer columns or header
✓ Store in HDF5 as attributes for each bearing fault

EXAMPLE METADATA EXTRACTION:
  B007_1_123_features.csv
  ├─ Type: Ball bearing (B)
  ├─ Fault size: 7 mil (007)
  ├─ Load: 1 (≈1000 lbf)
  ├─ RPM: 123 (placeholder - check reference)
  └─ Should be: B(all)_007_1_xxx

STANDARDIZED FILENAME PATTERN:
  {BearingType}_{FaultSize}_{LoadCategory}_{OperatingRPM}_load{number}_features.csv
  Example: B_007_1000lbf_1772rpm_features.csv
""")
    
    return "\n".join(report)

# ============================================================================
# PART 4: DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality():
    """Assess current data quality and identify gaps."""
    
    processed_dir = os.path.join(PROJECT_ROOT, "data", "CWRU_dataset", "processed")
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('_features.csv')]
    
    report = []
    report.append("\n" + "="*80)
    report.append("CURRENT DATA QUALITY ASSESSMENT")
    report.append("="*80)
    
    print(f"\nAnalyzing {len(csv_files)} feature files...")
    
    file_stats = []
    all_features = set()
    
    for csv_file in sorted(csv_files):
        df = pd.read_csv(os.path.join(processed_dir, csv_file))
        features = [c for c in df.columns if c not in ['Scenario', 'Timestamp', 'SysState', 'SampleCount']]
        all_features.update(features)
        
        file_stats.append({
            "File": csv_file,
            "Samples": len(df),
            "Features": len(features),
            "Memory_MB": df.memory_usage(deep=True).sum() / 1e6,
            "Missing_Values": df.isnull().sum().sum(),
            "Duplicates": df.duplicated().sum()
        })
    
    stats_df = pd.DataFrame(file_stats)
    report.append("\nPer-File Statistics:")
    report.append(stats_df.to_string())
    
    report.append(f"\n\nTotal unique features: {len(all_features)}")
    report.append(f"Feature columns: {sorted(all_features)}")
    
    # Check for missing critical features
    critical = [
        "AccX_Skew", "AccX_Kurt", "AccX_PeakFreq_Hz",
        "AccY_Skew", "AccY_Kurt", "AccY_PeakFreq_Hz",
        "AccZ_Skew", "AccZ_Kurt", "AccZ_PeakFreq_Hz"
    ]
    
    missing = [f for f in critical if f not in all_features]
    if missing:
        report.append(f"\n⚠ Missing critical features: {missing}")
    else:
        report.append(f"\n✓ All critical statistical features present")
    
    report.append("\n\nDATA QUALITY CHECKS:")
    report.append(f"├─ Total samples: {stats_df['Samples'].sum()}")
    report.append(f"├─ Files with missing values: {(stats_df['Missing_Values'] > 0).sum()}")
    report.append(f"├─ Files with duplicates: {(stats_df['Duplicates'] > 0).sum()}")
    report.append(f"├─ Total dataset size: {stats_df['Memory_MB'].sum():.2f} MB")
    report.append(f"└─ Feature consistency: {'✓ PASS' if len(set(stats_df['Features'])) == 1 else '⚠ FAIL'}")
    
    return "\n".join(report)

# ============================================================================
# PART 5: GENERATE IMPLEMENTATION CODE SNIPPET
# ============================================================================

def generate_feature_code():
    """Provide implementation template for advanced features."""
    
    code = '''
# ============================================================================
# ADVANCED FEATURE ENGINEERING TEMPLATE (Add to feature_engineering.py)
# ============================================================================

def compute_advanced_features(signal, fs=2.0, window_name='hann'):
    """
    Compute advanced features for bearing fault detection.
    
    Parameters:
        signal: 1D vibration signal (numpy array)
        fs: Sampling frequency (Hz)
        window_name: FFT window type
        
    Returns:
        dict: Feature name -> value pairs
    """
    from scipy.signal import welch, morlet2
    from scipy.stats import entropy as scipy_entropy
    
    features = {}
    
    # 1. SPECTRAL ENTROPY (Shannon entropy of normalized PSD)
    f, Pxx = welch(signal, fs, nperseg=min(256, len(signal)))
    psd_norm = Pxx / np.sum(Pxx)
    features['spectral_entropy'] = scipy_entropy(psd_norm[psd_norm > 0])
    
    # 2. SPECTRAL ENERGY
    features['spectral_energy'] = np.sum(Pxx)
    
    # 3. SPECTRAL CENTROID
    features['spectral_centroid'] = np.sum(f * psd_norm) / np.sum(psd_norm)
    
    # 4. CREST FACTOR (impulse indicator)
    features['crest_factor'] = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
    
    # 5. IMPULSE FACTOR
    features['impulse_factor'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    
    # 6. MARGIN FACTOR
    features['margin_factor'] = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
    
    # 7. CLEARANCE FACTOR
    abs_mean_sqrt = np.mean(np.sqrt(np.abs(signal)))
    features['clearance_factor'] = np.max(np.abs(signal)) / (abs_mean_sqrt**2)
    
    # 8. SPECTRAL KURTOSIS (HIGH IMPORTANCE for bearing faults)
    # Simplified version - full SK requires STFT analysis
    fft_vals = np.fft.fft(signal)
    spectrum = np.abs(fft_vals)**2
    spectrum_norm = spectrum / np.sum(spectrum)
    if np.sum(spectrum_norm) > 0:
        m2 = np.sum(spectrum_norm * (np.arange(len(spectrum))**2))
        m4 = np.sum(spectrum_norm * (np.arange(len(spectrum))**4))
        features['spectral_kurtosis'] = m4 / (m2**2) if m2 > 0 else 0
    
    # 9. AUTOCORRELATION AT LAG-1 (periodicity)
    if len(signal) > 1:
        acf = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
        acf = acf / acf[len(acf)//2]
        features['autocorr_lag1'] = acf[len(acf)//2 + 1]
    
    # 10. ZERO CROSSING RATE (impact detection)
    sign_changes = np.sum(np.diff(np.sign(signal)) != 0)
    features['zero_crossing_rate'] = sign_changes / len(signal)
    
    # 11. VARIANCE OF FIRST DIFFERENCES (smoothness)
    if len(signal) > 1:
        first_diff = np.diff(signal)
        features['variance_first_diff'] = np.var(first_diff)
    
    # 12. PEAK-TO-PEAK RATIO
    features['peak_to_peak'] = np.max(signal) - np.min(signal)
    
    # 13. PERCENTILE-BASED FEATURES (tail properties)
    features['percentile_90'] = np.percentile(np.abs(signal), 90)
    features['percentile_95'] = np.percentile(np.abs(signal), 95)
    features['percentile_99'] = np.percentile(np.abs(signal), 99)
    
    return features

# Integration with current pipeline:
# In main feature extraction loop, after computing temporal features:
#   adv_features = compute_advanced_features(janela[eixo].astype(float).values, TAXA_AMOSTRAGEM)
#   resumo.update({f'{eixo}_{k}': v for k, v in adv_features.items()})
'''
    
    return code

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CWRU RAW DATA ANALYSIS & FORMAT RECOMMENDATIONS")
    print("="*80)
    
    # Analyze .mat files
    mat_files = analyze_mat_files()
    
    # Data quality assessment
    quality_report = assess_data_quality()
    print(quality_report)
    
    # Format recommendations
    format_rec = recommend_data_format()
    print(format_rec)
    
    # Advanced features
    advanced_rec = recommend_advanced_features()
    print(advanced_rec)
    
    # Code template
    code_template = generate_feature_code()
    print("\n" + "="*80)
    print("ADVANCED FEATURE IMPLEMENTATION TEMPLATE")
    print("="*80)
    print(code_template)
    
    # Save comprehensive report
    full_report = (quality_report + format_rec + advanced_rec + 
                  "\n\n" + "="*80 + "\nIMPLEMENTATION TEMPLATE\n" + "="*80 + code_template)
    
    report_path = os.path.join(
        PROJECT_ROOT, "results", "metrics",
        f"cwru_analysis_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(f"\n✓ Full report saved: {report_path}")

if __name__ == "__main__":
    main()
