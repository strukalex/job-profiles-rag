# new module: app/profiles/handlers.py
from typing import List, Optional
import os
import time
from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import asyncio



# Initialize components
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(script_dir, "../..", "job_profiles_db")

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

async def get_context(query: str, k: int = 3):
    """Retrieve contextual documents from vector store"""
    loop = asyncio.get_event_loop()
    context_docs = await loop.run_in_executor(
        None,
        lambda: vectorstore.similarity_search(query, k)
    )
    return "\n\n".join([doc.page_content for doc in context_docs])

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
