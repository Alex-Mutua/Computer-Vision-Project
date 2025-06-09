import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os

def generate_heatmap(output_dir="outputs", target_class=None):
    """Generate heatmap from position_log.csv."""
    try:
        input_path = os.path.join(output_dir, "position_log.csv")
        output_path = os.path.join(output_dir, "heatmap.png")
        video_path = os.path.join(output_dir, "processed_video.mp4")

        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            return False

        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
            return False
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        positions = []
        with open(input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if target_class and row["class"].lower() != target_class.lower():
                    continue
                try:
                    x_center = (int(row["x_min"]) + int(row["x_max"])) / 2
                    y_center = (int(row["y_min"]) + int(row["y_max"])) / 2
                    positions.append([x_center, y_center])
                except (KeyError, ValueError) as e:
                    print(f"Error reading row {row}: {e}")
                    continue

        if not positions:
            print("No valid positions found in position_log.csv")
            return False

        positions = np.array(positions)
        bins = [int(width/20), int(height/20)]
        heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=bins)

        plt.figure(figsize=(width/100, height/100))
        sns.heatmap(heatmap.T, cmap="YlOrRd", cbar=True)
        plt.title(f"Heatmap of {target_class or 'All Objects'} Positions")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.xlim(0, width)
        plt.ylim(height, 0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Heatmap generated: {output_path}")
        return True
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return False

if __name__ == "__main__":
    generate_heatmap()