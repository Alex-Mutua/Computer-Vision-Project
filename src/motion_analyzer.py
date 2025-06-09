import pandas as pd
import numpy as np
def analyze_motion(position_log_path):
    try:
        df = pd.read_csv(position_log_path)
        if df.empty:
            return None
        # Compute centroid
        df["x_center"] = (df["x_min"] + df["x_max"]) / 2
        df["y_center"] = (df["y_min"] + df["y_max"]) / 2
        motion_data = []
        for track_id in df["track_id"].unique():
            track = df[df["track_id"] == track_id].sort_values("timestamp_ms")
            if len(track) < 2:
                continue
            # Calculate direction and speed
            for i in range(1, len(track)):
                dx = track.iloc[i]["x_center"] - track.iloc[i-1]["x_center"]
                dy = track.iloc[i]["y_center"] - track.iloc[i-1]["y_center"]
                dt = (track.iloc[i]["timestamp_ms"] - track.iloc[i-1]["timestamp_ms"]) / 1000  # seconds
                if dt == 0:
                    continue
                # Direction (angle in degrees)
                angle = np.arctan2(dy, dx) * 180 / np.pi
                direction = "right" if 45 <= angle < 135 else "down" if 135 <= angle < 225 else \
                            "left" if 225 <= angle < 315 else "up"
                # Speed (pixels/second)
                speed = np.sqrt(dx**2 + dy**2) / dt
                motion_data.append({
                    "track_id": track_id,
                    "class": track.iloc[i]["class"],
                    "timestamp_ms": track.iloc[i]["timestamp_ms"],
                    "direction": direction,
                    "speed_pps": speed
                })
        return pd.DataFrame(motion_data)
    except Exception as e:
        print(f"Error analyzing motion: {e}")
        return None