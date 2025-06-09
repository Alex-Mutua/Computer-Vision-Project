import cv2 as cv
import csv
import argparse
import os
from ultralytics import YOLO

class Detector:
    def __init__(self, filepath, target_object, output_dir="outputs", no_window=False):
        self.filepath = filepath
        self.target_object = target_object.lower()
        self.output_dir = output_dir
        self.model_path = "resources/models/yolo11n.pt"
        self.model = YOLO(self.model_path)
        self.no_window = no_window
        self.valid_classes = [
            "traffic light", "police car", "car",
            "person", "bus", "truck", "bicycle", "motorcycle"
        ]
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, confidence=0.25):
        cap = cv.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Erreur : impossible de lire la vidéo {self.filepath}")

        w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
        video_writer = cv.VideoWriter(
            os.path.join(self.output_dir, "processed_video.mp4"),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
        if not video_writer.isOpened():
            raise RuntimeError("Failed to initialize video writer")

        if not self.no_window:
            try:
                cv.namedWindow("Détection et Tracking", cv.WINDOW_NORMAL)
            except cv.error as e:
                print(f"OpenCV Error: Failed to create window: {e}")

        output_path = os.path.join(self.output_dir, "position_log.csv")
        detection_count = 0
        with open(output_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max"])

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Fin de la vidéo ou erreur de lecture")
                    break

                results = self.model.track(frame, persist=True, verbose=True, conf=confidence)[0]
                timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
                detected = False

                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id].lower()
                        if class_name in self.valid_classes and (self.target_object == "all" or class_name == self.target_object):
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            label = f"ID: {track_id} {class_name}"
                            cv.putText(frame, label, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            writer.writerow([timestamp, track_id, class_name, x_min, y_min, x_max, y_max])
                            detection_count += 1
                            detected = True

                if detected:
                    cv.putText(frame, f"{self.target_object.capitalize()} détecté !", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv.putText(frame, f"Aucun {self.target_object} détecté", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                video_writer.write(frame)
                if not self.no_window:
                    try:
                        cv.imshow("Détection et Tracking", frame)
                    except cv.error as e:
                        print(f"OpenCV Error: Failed to display frame: {e}")

                if cv.waitKey(10) & 0xFF == ord('q'):
                    print("Arrêt par l'utilisateur.")
                    break

        cap.release()
        video_writer.release()
        if not self.no_window:
            try:
                cv.destroyAllWindows()
            except cv.error as e:
                print(f"OpenCV Error: Failed to close windows: {e}")

        print(f"Detections written to {output_path}: {detection_count} total detections")

def parse_args():
    parser = argparse.ArgumentParser(description="Détection des objets avec YOLOv11")
    parser.add_argument("--filepath", type=str, required=True, help="Chemin de la vidéo")
    parser.add_argument("--target", type=str, default="all", help="Objet à détecter")
    parser.add_argument("--confidence", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--no-window", action="store_true", help="Désactiver la fenêtre OpenCV")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    detector = Detector(args.filepath, args.target, args.output_dir, args.no_window)
    detector.process_video(confidence=args.confidence)

if __name__ == "__main__":
    main()