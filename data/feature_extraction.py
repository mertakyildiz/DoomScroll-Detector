import numpy as np
import pandas as pd

# 1. Load your clean data
df = pd.read_csv("all_sessions_merged.csv")

# Configuration (100Hz data = 100 samples per second)
FS = 100  
WINDOW_DURATION = 4  # seconds
OVERLAP_PERCENTAGE = 0.50

# Convert configuration to number of rows/samples
window_size = int(WINDOW_DURATION * FS)  # 400 samples
step_size = int(window_size * (1 - OVERLAP_PERCENTAGE))  # 200 samples shift

# Define which numeric columns we want to extract features from
feature_cols = [
    "accelerometer_x", "accelerometer_y", "accelerometer_z",
    "gyroscope_x", "gyroscope_y", "gyroscope_z",
    "orientation_roll", "orientation_pitch", "orientation_yaw"
]

all_window_features = []

# 2. Process session-by-session so windows don't blur across different recordings
for session_id, session_df in df.groupby("session_id"):
    session_df = session_df.reset_index(drop=True)
    n_samples = len(session_df)
    
    session_label = session_df["label"].iloc[0] if "label" in session_df.columns else "unknown"

    # 3. Slide the window across the session data
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = session_df.iloc[start_idx:end_idx][feature_cols]
        
        # Dictionary to hold all features for this single window
        window_feat = {
            "session_id": session_id,
            "window_start_idx": start_idx,
            "label": session_label
        }
        
        # --- ROBUST FEATURE EXTRACTION BLOCK ---
        
        # A. Vector Magnitudes (Pure force/rotation strength, ignores device orientation angles)
        acc_mag = np.sqrt(window_data["accelerometer_x"]**2 + window_data["accelerometer_y"]**2 + window_data["accelerometer_z"]**2)
        gyro_mag = np.sqrt(window_data["gyroscope_x"]**2 + window_data["gyroscope_y"]**2 + window_data["gyroscope_z"]**2)
        
        window_feat["acc_mag_mean"] = acc_mag.mean()
        window_feat["acc_mag_std"] = acc_mag.std()
        window_feat["gyro_mag_mean"] = gyro_mag.mean()
        window_feat["gyro_mag_std"] = gyro_mag.std()

        # B. Robust metrics per individual axis
        for col in feature_cols:
            signal = window_data[col]
            signal_mean = signal.mean()
            
            # Standard Deviation (AC component tracker)
            window_feat[f"{col}_std"] = signal.std()
            
            # Interquartile Range (IQR - spreads out center 50% variance, ignoring harsh outlier spikes)
            q75, q25 = np.percentile(signal, [75 ,25])
            window_feat[f"{col}_iqr"] = q75 - q25
            
            # Mean Absolute Deviation (MAD - robust measure of structural variability)
            window_feat[f"{col}_mad"] = np.mean(np.abs(signal - signal_mean))
            
            # Zero Crossing Rate (ZCR - tracks frequency of micro-movements/shaking around the axis mean)
            zero_crossed = np.nonzero(np.diff(signal > signal_mean))[0]
            window_feat[f"{col}_zcr"] = len(zero_crossed) / window_size
            
        # C. Cross-Axis Correlations (Tracks coordinated gesture dynamics)
        window_feat["corr_acc_xy"] = window_data["accelerometer_x"].corr(window_data["accelerometer_y"])
        window_feat["corr_acc_yz"] = window_data["accelerometer_y"].corr(window_data["accelerometer_z"])
        window_feat["corr_gyro_xy"] = window_data["gyroscope_x"].corr(window_data["gyroscope_y"])
        
        # Handle potential NaNs in correlation if a sensor had absolute zero variance
        for k in ["corr_acc_xy", "corr_acc_yz", "corr_gyro_xy"]:
            if np.isnan(window_feat[k]):
                window_feat[k] = 0.0

        all_window_features.append(window_feat)

# 5. Combine into a final machine learning-ready DataFrame
features_df = pd.DataFrame(all_window_features)
features_df.to_csv("har_features.csv", index=False)

print(f"Extraction complete! Created {len(features_df)} windows.")
print(f"Features dataset shape: {features_df.shape}")