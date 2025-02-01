
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import torch
import os

class CrowdDetector:
    def __init__(self, crowd_threshold: int = 50, output_dir: str = "outputs"):
        """Initialize the crowd detector with YOLO model."""
        # Initialize YOLO model
        self.model = YOLO("yolov8x.pt")
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analytics parameters
        self.crowd_threshold = crowd_threshold
        self.alert_log = []
        self.person_count = 0
        self.track_history = {}
        self.suspicious_objects = ['knife', 'gun', 'baseball bat']
        
        # Behavior analysis parameters
        self.speed_threshold = 50
        self.proximity_threshold = 100

    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """Detect people in frame using YOLO."""
        try:
            results = self.model(frame, classes=[0])  # Only detect people
            detections = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'confidence': confidence,
                        'id': len(detections) + 1
                    }
                    detections.append(detection)
            
            self.person_count = len(detections)
            return detections
        except Exception as e:
            print(f"Error in detect_people: {str(e)}")
            return []

    def analyze_behavior(self, detections: List[Dict]) -> List[Dict]:
        """Analyze movement patterns and generate behavior alerts."""
        behaviors = []
        
        for det in detections:
            det_id = det['id']
            center = det['center']
            
            if det_id not in self.track_history:
                self.track_history[det_id] = []
            
            history = self.track_history[det_id]
            history.append(center)
            
            if len(history) > 30:  # Keep last 30 frames
                history.pop(0)
            
            if len(history) >= 2:
                # Calculate speed
                prev_pos = np.array(history[-2])
                curr_pos = np.array(history[-1])
                speed = np.linalg.norm(curr_pos - prev_pos)
                
                if speed > self.speed_threshold:
                    behaviors.append({
                        'type': 'RAPID_MOVEMENT',
                        'id': det_id,
                        'speed': float(speed),
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Check proximity
                for other_det in detections:
                    if other_det['id'] != det_id:
                        other_center = np.array(other_det['center'])
                        distance = np.linalg.norm(curr_pos - other_center)
                        if distance < self.proximity_threshold:
                            behaviors.append({
                                'type': 'CLOSE_PROXIMITY',
                                'id': det_id,
                                'other_id': other_det['id'],
                                'distance': float(distance),
                                'timestamp': datetime.now().isoformat()
                            })
        
        return behaviors

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a single frame and return detection results."""
        if frame is None:
            return None

        try:
            detections = self.detect_people(frame)
            behaviors = self.analyze_behavior(detections)
            
            alerts = []
            if len(detections) > self.crowd_threshold:
                alerts.append({
                    'type': 'OVERCROWDING',
                    'count': len(detections),
                    'threshold': self.crowd_threshold,
                    'timestamp': datetime.now().isoformat()
                })
            
            alerts.extend(behaviors)
            self.alert_log.extend(alerts)
            
            return {
                'detections': detections,
                'behaviors': behaviors,
                'alerts': alerts,
                'count': len(detections)
            }
        
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on frame."""
        output = frame.copy()
        
        # Draw detections
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            det_id = det['id']
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"ID: {det_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw trajectory
            if det_id in self.track_history:
                points = np.array(self.track_history[det_id], dtype=np.int32)
                if len(points) >= 2:
                    cv2.polylines(output, [points.reshape((-1, 1, 2))], 
                                False, (0, 255, 255), 2)
        
        # Draw count and alerts
        cv2.putText(output, f"People Count: {results['count']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        y_offset = 60
        for alert in results['alerts']:
            color = (0, 0, 255) if alert['type'] in ['RAPID_MOVEMENT', 'CLOSE_PROXIMITY'] else (0, 165, 255)
            cv2.putText(output, f"ALERT: {alert['type']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        return output

    def process_video(self, video_path: str) -> str:
        """Process video file and save results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = str(self.output_dir / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.process_frame(frame)
                if results:
                    processed_frame = self.draw_results(frame, results)
                    writer.write(processed_frame)
                    
                    # Show frame while processing
                    cv2.imshow('Processing', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
        
        return output_path

    def get_analytics(self) -> Dict[str, Any]:
        """Get current analytics data."""
        return {
            'total_alerts': len(self.alert_log),
            'recent_alerts': self.alert_log[-5:],
            'timestamp': datetime.now().isoformat()
        }