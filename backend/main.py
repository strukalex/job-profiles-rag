from dotenv import load_dotenv
# Load environment variables
load_dotenv()

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
            "id": "job-profile-rag",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "your-company"
        }]
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
