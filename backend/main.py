from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import cv2
import asyncio
import threading
from datetime import datetime
import time
from detection import CrowdDetector

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_broadcast_time = 0
        self.broadcast_interval = 2.0  # 2 seconds interval

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New connection. Total active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Connection closed. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        current_time = time.time()
        if current_time - self.last_broadcast_time < self.broadcast_interval:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to connection: {str(e)}")
                disconnected.append(connection)
        
        for conn in disconnected:
            await self.disconnect(conn)
            
        self.last_broadcast_time = current_time

app = FastAPI(title="Real-Time Crowd Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global objects
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

detector = CrowdDetector(crowd_threshold=50)
manager = ConnectionManager()

async def process_video_frame(frame):
    try:
        results = detector.process_frame(frame)
        if results and results.get('alerts'):
            await manager.broadcast({
                'type': 'ALERT',
                'data': results['alerts'],
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

def process_video_thread(video_path: str):
    cap = cv2.VideoCapture(video_path)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            asyncio.run(process_video_frame(frame))
            
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
    finally:
        cap.release()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
            analytics = detector.get_analytics()
            await websocket.send_json(analytics)
    except Exception:
        await manager.disconnect(websocket)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        thread = threading.Thread(target=process_video_thread, args=(file_path,))
        thread.start()
        
        return JSONResponse(content={
            "message": "Video processing started",
            "video_file": file_path
        }, status_code=200)
    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)

@app.get("/analytics")
async def get_analytics():
    return detector.get_analytics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)