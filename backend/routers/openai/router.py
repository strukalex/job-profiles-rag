 

# updated original module (router.py)
import os
import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from backend.handlers.classify_profile import handle_classify_profile
from backend.handlers.draw_pr_graph import handle_draw_pr_graph
from backend.handlers.draw_profiles_graph import handle_draw_profile_graph
from backend.handlers.provide_help import handle_provide_help
from backend.handlers.provide_self_help import handle_provide_self_help
from handlers.generate_profile import handle_generate_profile
from handlers.profiles import handle_profile_analysis
from ..semantic.layer import route_layer

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 300

@router.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    query = next((msg.content for msg in reversed(request.messages) 
                 if msg.role == "user"), "")
    
    last_message = request.messages[-1].content
    route = route_layer.route(last_message)

    print('ROUTING TO: ', route)

    if route.name == "profile_search":
        return await handle_profile_analysis(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "generate_profile":
        return await handle_generate_profile(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "draw_profiles_graph":
        return await handle_draw_profile_graph(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "draw_pr_graph":
        return await handle_draw_pr_graph(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "classify_profile":
        return await handle_classify_profile(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "provide_help":
        return await handle_provide_help(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    elif route.name == "provide_self_help":
        return await handle_provide_self_help(
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