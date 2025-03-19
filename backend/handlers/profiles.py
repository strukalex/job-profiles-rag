# new module: app/profiles/handlers.py
from typing import List, Optional
import os
import time
import json
import uuid
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
import asyncio
from langchain_core.prompts import ChatPromptTemplate

from .azure_client import azure_client, get_langchain_azure_model


# Initialize components
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(script_dir, "../..", "job_profiles_db2")

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="job_profiles"
)

# Use the encoder from our Azure client wrapper
encoder = MistralTokenizer.from_model("mistral-small", strict=True)

def count_tokens(text: str) -> int:
    return azure_client.count_tokens(text)

def print_document_simple(doc):
    """Print document details"""
    print("\n DOCUMENTS: " + "="*80)
    print("-"*80)
    print(doc.page_content)
    print("="*80 + "\n")

async def get_context(query: str, k: int = 3):
    """Retrieve contextual documents from vector store with token limiting"""
    loop = asyncio.get_event_loop()
    context_docs = await loop.run_in_executor(
        None,
        lambda: vectorstore.similarity_search(
            'Accountabilities/Education/Job Experience/Professional Registration Requirements/Willingness Statements/Security Screenings' + query,
            k=20
        )
    )

    total_tokens = 0
    processed_docs = []
    
    for idx, document in enumerate(context_docs):
        print_document_simple(document)
        doc_tokens = count_tokens(document.page_content)
        
        if total_tokens + doc_tokens <= 3500:
            processed_docs.append(document.page_content)
            total_tokens += doc_tokens
        else:
            print(f'TRUNCATING CONTEXT at {total_tokens} tokens k={idx}')
            break

    context_string = "\n\n".join(processed_docs)
    token_count = count_tokens(context_string)
    print('context token length: ', token_count)
    
    return context_string


class ProfileAnalyzer:
    def __init__(self):
        self.llm = get_langchain_azure_model(
            model_name=os.getenv('MODEL_NAME'),
            api_version="2024-05-01-preview",
            model_kwargs={
                "max_tokens": 8000,
                "stream": True
            }
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """System: You're an expert in answering questions about a library of job profiles. Use this context:
            {context}

            User: {query}
            Assistant: """
        )
        
        self.chain = (
            self.prompt
            | self.llm
        )


_PROFILE_ANALYZER = ProfileAnalyzer()

async def handle_profile_analysis(
    query: str,
    model: str = os.getenv('MODEL_NAME'),
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    """Handle the profile analysis RAG workflow"""
    try:
        context = await get_context(query)
        
        print('context: ', context)
        
        # Use the chain approach like provide_help.py
        response = await _PROFILE_ANALYZER.chain.ainvoke({
            "context": context,
            "query": query
        })

        if isinstance(response, JSONResponse):
            return response
    
        return {
            "id": "chatcmpl-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content
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

async def handle_profile_analysis_stream(
    query: str,
    model: str = os.getenv('MODEL_NAME'),
    temperature: float = 0.7,
    max_tokens: int = 300
):
    """Handle the profile analysis RAG workflow with streaming"""
    try:
        context = await get_context(query)
        
        print('context: ', context)
        
        # Create an async generator that yields formatted chunks
        async def generate_stream():
            # Create response header in OpenAI format
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())
            
            # Send the initial response data
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Stream the response content using the chain like in provide_help.py
            async for chunk in _PROFILE_ANALYZER.chain.astream({
                "context": context,
                "query": query
            }):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = chunk
                    
                # Format each chunk in OpenAI's streaming format
                json_data = {
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model,
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'content': content},
                            'finish_reason': None
                        }
                    ]
                }
                # Then format the string
                yield f"data: {json.dumps(json_data)}\n\n"
            
            # Send the final [DONE] message
            final_json_data = {
                'id': response_id,
                'object': 'chat.completion.chunk',
                'created': created_time,
                'model': model,
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop'
                    }
                ]
            }
            # Then format the string
            yield f"data: {json.dumps(final_json_data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        # Return a streaming response
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
