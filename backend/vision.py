import requests
import cv2
import os

# Groq API Key (Replace with actual key)
GROQ_API_KEY = "your_groq_api_key"
GROQ_API_URL = "https://api.groq.com/v1/vision"


def analyze_crowd(frame):
    """
    Send a frame to LLaMA 3.2 Vision for AI analysis.

    Returns a text description of the scene.
    """
    # Save the frame temporarily
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    # Open the file
    with open(temp_frame_path, "rb") as image_file:
        files = {"image": image_file}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        data = {"prompt": "Describe the crowd density and any potential risks in this image."}

        response = requests.post(GROQ_API_URL, headers=headers, files=files, data=data)

    # Remove temporary file
    os.remove(temp_frame_path)

    # Return the AI's analysis
    if response.status_code == 200:
        return response.json().get("analysis", "No insights available.")
    else:
        return "Error analyzing crowd."
