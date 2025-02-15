from fastapi import APIRouter, FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from ..semantic.layer import route_layer
import asyncio
import time
import os

# Your existing imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 300

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(script_dir, "../../../", "job_profiles_db")

print('persist_dir: ', persist_dir)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="job_profiles"
)

client = ChatCompletionsClient(
    endpoint=os.getenv('AZURE_ENDPOINT'),
    credential=AzureKeyCredential(os.getenv('AZURE_API_KEY')),
    model="Mistral-small"
)

print('vectorstore: ', len(vectorstore.get()['documents']))

async def get_context(query: str, k: int = 3):
    loop = asyncio.get_event_loop()
    context_docs = await loop.run_in_executor(
        None,
        lambda: vectorstore.similarity_search(query, k)
    )
    return "\n\n".join([doc.page_content for doc in context_docs])

router = APIRouter()


async def handle_profile_analysis(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    """Handle the profile analysis RAG workflow"""
    try:
        context = await get_context(query)
        
        print('context: ', context)

        prompt_template = (
            f"System: You're an expert in answering questions about a library "
            f"of job profiles. Use this context:\n{context}\n\n"
            f"User: {query}\n"
            f"Assistant: "
        )
        
        response = client.complete(
            messages=[{"role": "user", "content": prompt_template}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print('response: ', response)

        return {
            "id": "chatcmpl-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    query = next((msg.content for msg in reversed(request.messages) 
                 if msg.role == "user"), "")
    
    # Extract last user message
    last_message = request.messages[-1].content
    
    # Use semantic routing to determine handler
    route = route_layer.route(last_message)

    print('route: ', route)

    # Your routing logic would go here
    if route.name=="profile_search":  # Implement your routing logic
        return await handle_profile_analysis(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    else:
        return {
            "id": "chatcmpl-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sorry, I don't know how to do that."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    