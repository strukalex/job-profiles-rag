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
from fastapi import FastAPI, HTTPException, Body, Depends, Security
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn
from routers.openai.router import router as openai_router
from handlers.token_limiter import token_limiter
from handlers.azure_client import azure_client
from utils.logging_config import setup_logging
from auth.middleware import AuthMiddleware
from auth.api_key import get_current_user, require_admin, security_scheme, User
from starlette.middleware.base import BaseHTTPMiddleware

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

app = FastAPI(
    title="JobStore AI API",
    description="API for JobStore AI with API Key Authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Custom OpenAPI schema to improve authentication documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="JobStore AI API",
        version="1.0.0",
        description="""
        # API Key Authentication
        
        This API uses Bearer token authentication. To authenticate:
        
        1. Click the 'Authorize' button
        2. In the 'Bearer Authentication' section, enter your API key (without 'Bearer ' prefix)
        3. Click 'Authorize' and then 'Close'
        
        All authenticated requests will now include your API key as a Bearer token.
        """,
        routes=app.routes,
    )
    
    # Define security scheme
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    
    # Add Bearer token security scheme
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "API Key",
        "description": "Enter your API key as the token value"
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define public paths that don't require authentication
public_paths = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/static"
]

# Add authentication middleware as a regular middleware
app.add_middleware(AuthMiddleware, public_paths=public_paths)

# Mount OpenAI-compatible endpoints
app.include_router(openai_router)

@app.get("/v1/models")
async def list_models(user: User = Security(get_current_user)):
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
async def token_usage(user: User = Security(require_admin)):
    """Get current token usage statistics. Requires admin access."""
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
