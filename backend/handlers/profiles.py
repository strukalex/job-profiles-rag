# new module: app/profiles/handlers.py
from typing import List, Optional
import os
import time
from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
import asyncio



# Initialize components
script_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(script_dir, "../..", "job_profiles_db2")

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
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

encoder = MistralTokenizer.from_model("mistral-small", strict=True)

def count_tokens(text: str) -> int:
    return len(encoder.encode_chat_completion(
        ChatCompletionRequest(
            messages=[UserMessage(content=text)],
            model="mistral-small-latest"
        )
    ).tokens)

def print_document_simple(doc):
    """Print document details"""
    print("\n DOCUMENT: " + "="*80)
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
