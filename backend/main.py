from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from detection import detect_people

# Initialize FastAPI app
app = FastAPI(
    title="Crowd Management API",
    description="API for processing videos with YOLOv11 for crowd detection",
    version="1.0"
)

# Create upload directory if it doesn’t exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload_video/", summary="Upload a video for YOLOv11 processing")
async def upload_video(file: UploadFile = File(...)):
    """
    Uploads a video file and processes it using YOLOv11.

    **Returns**:
    - Processed video file path
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded video
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Process the video with YOLOv11
    output_video_path = detect_people(file_path)

    return JSONResponse(
        content={
            "message": "Processing complete",
            "output_video": output_video_path
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
