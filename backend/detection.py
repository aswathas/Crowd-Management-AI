import cv2
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8x.pt")  # Use the latest YOLO model


def detect_people(frame, model):
    """Detect people in the given frame using YOLOv8"""

    # Preprocess the frame (if needed)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = model(frame_rgb)  # YOLO model inference

    if results is None or not results:  # Check for empty results
        return frame, 0

    people_count = 0

    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue  # Skip if no detections

        for box in result.boxes:
            if not (hasattr(box, 'xyxy') and hasattr(box, 'conf')):
                continue  # Skip invalid boxes

            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            people_count += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, people_count
