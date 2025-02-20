from pathlib import Path
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
# Load environment variables
load_dotenv()

# Add the project root to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from routers.openai.router import router as openai_router




app = FastAPI()

# Mount OpenAI-compatible endpoints
app.include_router(openai_router)

@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": "JobStore AI",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "your-company"
        }]
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Create and mount static directory
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
