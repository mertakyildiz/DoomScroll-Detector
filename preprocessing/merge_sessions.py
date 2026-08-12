import json
import glob
import pandas as pd

files = glob.glob("records/*.json")
all_sessions = []

SENSORS = {
    "TotalAcceleration": ["x", "y", "z"],
    "Gyroscope":         ["x", "y", "z"],
    "Orientation":       ["roll", "pitch", "yaw"],
}

SENSOR_PREFIX = {
    "TotalAcceleration": "accelerometer",
    "Gyroscope":         "gyroscope",
    "Orientation":       "orientation",
}

for filepath in files:
    data = json.load(open(filepath))
    if not isinstance(data, list) or not isinstance(data[0], dict):
        continue

    session_id = filepath.replace(".json", "")
    tags = [r["tag"] for r in data if r.get("sensor") == "Tags"]
    label = ", ".join(tags) if tags else "unknown"

    # Build one DataFrame per sensor
    sensor_dfs = {}
    for sensor, cols in SENSORS.items():
        rows = [r for r in data if r.get("sensor") == sensor]
        df = pd.DataFrame(rows)
        df["time_ns"] = df["time"].astype(float)
        df = df.set_index("time_ns")[cols].astype(float)
        df.columns = [f"{SENSOR_PREFIX[sensor]}_{c}" for c in cols]
        sensor_dfs[sensor] = df

    # Common 100Hz grid (10ms = 10_000_000 ns) spanning the session
    t_start = min(df.index.min() for df in sensor_dfs.values())
    t_end   = max(df.index.max() for df in sensor_dfs.values())
    grid = pd.DataFrame(index=pd.RangeIndex(int(t_start), int(t_end), 10_000_000))
    grid.index.name = "time_ns"

    # Align each sensor to the grid via interpolation.
    # grid.join() only matches exact timestamps; instead we merge the sensor
    # timestamps into the grid index, interpolate between readings, then
    # project back to the 10ms grid points.
    aligned = []
    for df in sensor_dfs.values():
        combined_idx = df.index.union(grid.index)
        resampled = (df.reindex(combined_idx)
                       .interpolate(method="index")
                       .bfill()
                       .reindex(grid.index))
        aligned.append(resampled)

    combined = pd.concat(aligned, axis=1)
    combined.insert(0, "session_id", session_id)
    combined.insert(1, "label", label)
    all_sessions.append(combined)

result = pd.concat(all_sessions)
result.index.name = "time_ns"
result.to_csv("all_sessions_merged.csv")
print(f"Done: {len(result)} rows, {len(all_sessions)} session(s) -> merged.csv")