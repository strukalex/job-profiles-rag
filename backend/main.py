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
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
from routers.openai.router import router as openai_router
from handlers.token_limiter import token_limiter
from handlers.azure_client import azure_client
from utils.logging_config import setup_logging

from langchain.globals import set_debug
set_debug(True)

# Set up logging using centralized configuration
setup_logging()
logger = logging.getLogger('main')

# Initialize token usage data
try:
    stats = token_limiter.get_usage_stats()
    logger.info(f"Token usage initialized: {stats}")
except Exception as e:
    logger.error(f"Error initializing token usage: {e}")

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

@app.get("/token-usage")
async def token_usage():
    """Get current token usage statistics."""
    try:
        return token_limiter.get_usage_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get the directory where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Create and mount static directory
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
