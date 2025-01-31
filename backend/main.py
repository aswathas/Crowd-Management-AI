from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import threading
from detection import detect_people

app = FastAPI(title="Real-Time Crowd Management API", version="2.1")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video and starts YOLOv8x processing."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save uploaded video
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        print(f"✅ Video saved: {file_path}")

        # Start YOLOv8x processing in a separate thread
        thread = threading.Thread(target=detect_people, args=(file_path,))
        thread.start()

        return JSONResponse(content={
            "message": "YOLOv8x processing started!",
            "video_file": file_path
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/processed_videos/")
async def list_processed_videos():
    """Lists all processed videos in the output directory."""
    videos = os.listdir(OUTPUT_DIR)
    return {"processed_videos": videos}


@app.get("/processed_videos/{filename}")
async def get_processed_video(filename: str):
    """Retrieves a processed video by filename."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return {"video_url": f"http://127.0.0.1:8000/output/{filename}"}
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
