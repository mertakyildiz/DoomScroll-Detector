import numpy as np
import pandas as pd

df = pd.read_csv("all_sessions_merged.csv")

FS = 100
WINDOW_DURATION = 4
OVERLAP_PERCENTAGE = 0.50
N_CHUNKS = 4

window_size = int(WINDOW_DURATION * FS)   # 400 samples
step_size   = int(window_size * (1 - OVERLAP_PERCENTAGE))  # 200 samples

feature_cols = [
    "accelerometer_x", "accelerometer_y", "accelerometer_z",
    "gyroscope_x",     "gyroscope_y",     "gyroscope_z",
    "orientation_roll","orientation_pitch","orientation_yaw",
]

freqs = np.fft.rfftfreq(window_size, d=1.0 / FS)


def add_fft_features(signal_vals, prefix, feat):
    fft_vals   = np.abs(np.fft.rfft(signal_vals))
    power      = fft_vals ** 2
    total_power = power.sum() + 1e-10
    p          = power / total_power

    feat[f"{prefix}_dom_freq"]     = float(freqs[1 + np.argmax(fft_vals[1:])])
    feat[f"{prefix}_spec_entropy"] = float(-np.sum(p * np.log(p + 1e-10)))
    feat[f"{prefix}_band_walk"]    = float(power[(freqs >= 1)   & (freqs < 3)].sum() / total_power)
    feat[f"{prefix}_band_gesture"] = float(power[(freqs >= 0.5) & (freqs < 2)].sum() / total_power)
    feat[f"{prefix}_band_fine"]    = float(power[(freqs >= 3)   & (freqs < 8)].sum() / total_power)


all_window_features = []

for session_id, session_df in df.groupby("session_id"):
    session_df   = session_df.reset_index(drop=True)
    n_samples    = len(session_df)
    session_label = session_df["label"].iloc[0] if "label" in session_df.columns else "unknown"
    n_windows    = max(1, (n_samples - window_size) // step_size + 1)

    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx     = start_idx + window_size
        window_data = session_df.iloc[start_idx:end_idx][feature_cols]

        window_num = start_idx // step_size
        chunk_id   = (window_num * N_CHUNKS) // n_windows

        feat = {
            "session_id":     session_id,
            "sub_session_id": f"{session_id}_c{chunk_id}",
            "window_start_idx": start_idx,
            "label": session_label,
        }

        # --- Vector magnitudes ---
        acc_mag  = np.sqrt(window_data["accelerometer_x"]**2 +
                           window_data["accelerometer_y"]**2 +
                           window_data["accelerometer_z"]**2)
        gyro_mag = np.sqrt(window_data["gyroscope_x"]**2 +
                           window_data["gyroscope_y"]**2 +
                           window_data["gyroscope_z"]**2)

        feat["acc_mag_mean"]  = float(acc_mag.mean())
        feat["acc_mag_std"]   = float(acc_mag.std())
        feat["gyro_mag_mean"] = float(gyro_mag.mean())
        feat["gyro_mag_std"]  = float(gyro_mag.std())

        add_fft_features(acc_mag.values,  "acc_mag",  feat)
        add_fft_features(gyro_mag.values, "gyro_mag", feat)

        # --- Per-axis features ---
        for col in feature_cols:
            vals        = window_data[col].values
            col_mean    = vals.mean()

            # Time-domain
            feat[f"{col}_std"] = float(vals.std())
            q75, q25 = np.percentile(vals, [75, 25])
            feat[f"{col}_iqr"] = float(q75 - q25)
            feat[f"{col}_mad"] = float(np.mean(np.abs(vals - col_mean)))
            zero_crossed = np.nonzero(np.diff(vals > col_mean))[0]
            feat[f"{col}_zcr"] = float(len(zero_crossed) / window_size)

            # Jerk (rate of change)
            jerk = np.diff(vals) * FS
            feat[f"{col}_jerk_std"]      = float(jerk.std())
            feat[f"{col}_jerk_mean_abs"] = float(np.abs(jerk).mean())

            # Frequency-domain
            add_fft_features(vals, col, feat)

        # --- Cross-axis correlations ---
        feat["corr_acc_xy"]  = float(window_data["accelerometer_x"].corr(window_data["accelerometer_y"]))
        feat["corr_acc_yz"]  = float(window_data["accelerometer_y"].corr(window_data["accelerometer_z"]))
        feat["corr_gyro_xy"] = float(window_data["gyroscope_x"].corr(window_data["gyroscope_y"]))

        for k in ["corr_acc_xy", "corr_acc_yz", "corr_gyro_xy"]:
            if np.isnan(feat[k]):
                feat[k] = 0.0

        all_window_features.append(feat)

features_df = pd.DataFrame(all_window_features)

# Convenience label columns for the two-step classifier.
# When the label is "unknown" (no Tags sensor in the JSON), fall back to the
# session path, which always encodes the activity (e.g. "Walking_scrolling").
# The regex handles the "scolling" typo present in some filenames.
label_or_path = features_df["label"].where(
    features_df["label"] != "unknown", features_df["session_id"]
)
features_df["is_walking"]   = label_or_path.str.contains("Walking",       case=False).astype(int)
features_df["is_scrolling"] = label_or_path.str.contains("scroll|scoll",  case=False, regex=True).astype(int)

features_df.to_csv("har_features.csv", index=False)

print(f"Extraction complete! {len(features_df)} windows, {features_df.shape[1]} columns.")
print(f"Sub-sessions: {features_df['sub_session_id'].nunique()} "
      f"({N_CHUNKS} chunks × {features_df['session_id'].nunique()} sessions)")
print(f"\nClass distribution:")
print(features_df.groupby(["is_walking", "is_scrolling"]).size().rename("windows"))
