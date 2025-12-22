import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import json
from typing import Dict

app = FastAPI(title="Simple ChartReader API", description="API to upload image, run val_extraction.py, and return JSON result.")

# Constants (adapt paths if needed)
DATA_DIR = "./data"  # Base dir for saving images
EVALUATION_DIR = "fast_api_evaluation"  # Updated base for save_path (from your log)
CACHE_PATH = "./cache"
MODEL_TYPE = "KPDetection"
TRAINED_ITER = "best"

# Ensure base dirs exist (optimized: makedirs if not, O(1))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

@app.post("/process_image/", response_model=Dict)
async def process_image(file: UploadFile):
    """Endpoint: Accept image, save to unique folder with required structure (images/val), run script, return JSON from save_path.
    Пояснение: Async for concurrency. UUID for unique names (collision-free, fast gen). Subprocess.run for script exec (shell=False for security, check=True for errors). JSON load after script (direct from file, O(n) by size)."""
    try:
        # Generate unique folder names (method: uuid4 — random 128-bit, <1ms)
        unique_id = str(uuid.uuid4())
        image_base_dir = os.path.join(DATA_DIR, unique_id)
        save_path = os.path.join(EVALUATION_DIR, unique_id)

        # Create save_path dir before script (method: makedirs recursive, exist_ok for idempotency — prevents FileNotFound)
        os.makedirs(save_path, exist_ok=True)

        # Create required structure: {data_dir}/images/val/{filename} (method: makedirs recursive, exist_ok for no errors)
        image_val_dir = os.path.join(image_base_dir, "images", "val")
        os.makedirs(image_val_dir, exist_ok=True)

        # Save image to val dir (method: contents to file, binary write for efficiency)
        file_path = os.path.join(image_val_dir, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Run script with data_dir=image_base_dir (script sees images/val)
        cmd = [
            "python", "val_extraction.py",
            "--save_path", save_path,
            "--model_type", MODEL_TYPE,
            "--cache_path", CACHE_PATH,
            "--data_dir", image_base_dir,
            "--trained_model_iter", TRAINED_ITER
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Script failed: {result.stderr}")

        # Find and load JSON (method: os.listdir for single file, assume one JSON per save_path. O(1) if few files)
        json_files = [f for f in os.listdir(save_path) if f.endswith('.json')]
        if not json_files:
            raise ValueError("No JSON found in save_path")

        json_path = os.path.join(save_path, json_files[0])  # Take first (assume one)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        return {"result": json_data, "save_path": save_path, "image_dir": image_base_dir}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script error: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)