

import json
import os
import time
# from azure.ai.inference import ChatCompletionsClient
# from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI, OpenAI, ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

class JobProfileGenerator:
    def __init__(self, csv_path, threshold=0.95):
        self.csv_path = csv_path
        self.threshold = threshold

        # It's not "AzureChatOpenAI", or "OpenAI", or "ChatCompletionsClient", it's...
        self.llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credential=os.getenv('AZURE_API_KEY'),
            model_name="Mistral-small",
            api_version="2024-05-01-preview"
        )

        self.embeddings =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize vector store with CSV data
        # self.loader = CSVLoader(file_path=csv_path)
        # self.documents = self._process_csv_documents()
        # self.vector_store = Chroma.from_documents(
        #     documents=self.documents,
        #     embedding=self.embeddings,
        #     collection_name="job_profiles"
        # )
        
        # Set up JSON generation chain
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """Generate a job profile in JSON format matching this structure:
            {{
                "title": string,
                "pay_grade": string,
                "accountabilities": [string],
                "requirements": [string],
                "experience_years": integer
            }}
            
            User request: {request}
            {format_instructions}
            Use exact terminology from industry standards where applicable."""
        )

        self.chain = self.prompt | self.llm | self.parser

    def generate_profile(self, request, phase=2):
        # Phase 1: Basic generation
        generated = self.chain.invoke({
            "request": request,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        if phase == 1:
            return generated
        
        # Phase 2: Semantic alignment with existing data
        # aligned = {}
        # for field, value in generated.items():
        #     if isinstance(value, list):
        #         aligned[field] = [
        #             self._find_exact_match(item, field) or item
        #             for item in value
        #         ]
        #     elif field == "experience_years":
        #         # Special handling for numerical fields
        #         existing = self._find_exact_match(str(value), "experience")
        #         aligned[field] = self._validate_experience(value, existing) if existing else value
        #     else:
        #         existing = self._find_exact_match(value, field)
        #         aligned[field] = existing or value
        
        # return aligned
        #     
    # def _process_csv_documents(self):
    #     """Convert CSV rows into individual documents for vector store"""
    #     docs = []
    #     raw_docs = self.loader.load()
        
    #     for doc in raw_docs:
    #         content = doc.page_content
    #         metadata = doc.metadata
    #         # Create separate documents for each field type
    #         docs.append(Document(
    #             page_content=f"Title: {json.loads(content)['title']}",
    #             metadata={"field": "title", **metadata}
    #         ))
    #         docs.append(Document(
    #             page_content=f"Pay Grade: {json.loads(content)['pay_grade']}",
    #             metadata={"field": "pay_grade", **metadata}
    #         ))
    #         for acc in json.loads(content)['accountabilities']:
    #             docs.append(Document(
    #                 page_content=f"Accountability: {acc}",
    #                 metadata={"field": "accountability", **metadata}
    #             ))
    #     return docs

    # def _find_exact_match(self, text, field_type):
    #     """Find best matching existing phrase using vector similarity"""
    #     results = self.vector_store.similarity_search_with_score(
    #         query=text,
    #         k=1,
    #         filter={"field": field_type}
    #     )
        
    #     if results and results[0][1] >= self.threshold:
    #         content = results[0][0].page_content
    #         # Extract just the value part after field prefix
    #         return content.split(": ", 1)[1]  
    #     return None

    # def _validate_experience(self, generated, existing):
    #     """Strict validation for numerical fields"""
    #     gen_num = int(re.search(r'\d+', generated).group())
    #     existing_num = int(re.search(r'\d+', existing).group())
    #     return existing if gen_num == existing_num else generated

gg = JobProfileGenerator("job_profiles.csv")

async def handle_generate_profile(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    generator = JobProfileGenerator("job_profiles.csv")

    # Phase 1 usage
    phase1_result = generator.generate_profile(
        "Senior Software Developer specializing in cloud infrastructure",
        phase=1
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
                    "content": json.dumps(phase1_result) #response.choices[0].message.content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }