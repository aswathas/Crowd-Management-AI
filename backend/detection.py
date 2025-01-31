import cv2
from ultralytics import YOLO
import os

# Load YOLOv8x Model
model = YOLO("yolov8x.pt")

MAX_SAFE_PEOPLE = 50
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists


def detect_people(video_path):
    """Detects people in a video, saves the processed video, and logs results."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = os.path.basename(video_path).replace(".mp4", "_processed.mp4")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        people_count = sum(len(result.boxes) for result in results)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if people_count > MAX_SAFE_PEOPLE:
            cv2.putText(frame, "🚨 ALERT: Overcrowding!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(frame)
        frame_count += 1
        print(f"Processing frame {frame_count}, People Count: {people_count}")

    cap.release()
    out.release()
    print(f"✅ Processed video saved at: {output_path}")
    return output_path
