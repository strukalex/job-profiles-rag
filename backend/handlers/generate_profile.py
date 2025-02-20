

import json
import os
import pprint
import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.structured_query import Comparator, Operator
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage

import tiktoken
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser, 
    get_query_constructor_prompt,
    AttributeInfo
)
import pandas as pd


from mistral_common.protocol.instruct.request import ChatCompletionRequest
from config.definitions import ROOT_DIR
from langchain_core.documents import Document

from langchain.globals import set_debug
set_debug(True)

array_fields={
            "accountabilities": "Accountabilities",
            "knowledge_skills_abilities": "Knowledge, Skills, and Abilities",
            "job_experience": "Job Experience",
            "education": "Education",
            "professional_registration_requirements": "Professional Registration Requirements",
            "preferences": "Preferences",
            "willingness_statements": "Willingness Statements",
            "security_screenings": "Security Screenings",
            "behavioural_competencies": "Behavioural Competencies",
        }

class JobProfileGenerator:
    def __init__(self, csv_path, threshold=0.17):
        self.csv_path = csv_path
        self.threshold = threshold

        # It's not "AzureChatOpenAI", or "OpenAI", or "ChatCompletionsClient", it's...
        self.llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credential=os.getenv('AZURE_API_KEY'),
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            model_kwargs={
                "max_tokens": 8000,
                # "stop": ["\n", "###"]  # Add natural stopping points
            }
        )

        

        self.encoder = MistralTokenizer.from_model("mistral-small", strict=True)

        self.embeddings =  HuggingFaceEmbeddings(model_name="thenlper/gte-small")
        #SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
        
        # Initialize vector store with CSV data
        
        DATA_DIR = ROOT_DIR / "data"
        fpath = DATA_DIR /csv_path
        self.loader = CSVLoader(file_path=fpath)
        self.documents = self._process_csv_with_pandas()
        print('creating vector store..')
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name="job_profiles"
        )
        print('done creating vector store..')

        # test code for vector store:
        # collection_data = self.vector_store.get()
        # results = self.vector_store.similarity_search("software engineer", k=3)[0]
        
        # Set up JSON generation chain
        self.parser = JsonOutputParser()
        # self.prompt = ChatPromptTemplate.from_template(
        #     """Generate a job profile in JSON format matching this structure:
        #     {{
        #         "title": string,
        #         "pay_grade": string,
        #         "accountabilities": [string]
        #     }}
            
        #     User request: {request}
        #     {format_instructions}
        #     Use exact terminology from industry standards where applicable."""
        # )

        self.rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job profile assistant. Use the following context to generate job profiles.
                     Only use information from the provided context."""),
        ("user", """Context: {context}
                    
                    User Request: {request}
                    
                    Generate a job profile in JSON format matching this structure. Only include fields if required:
                    {{
                        "title": string,
                        "accountabilities": [string],
                        "education": [string],
                        "job_experience": [string],
                        "professional_registration_requirements": [string],
                        "preferences": [string],
                        "knowledge_skills_abilities": [string],
                        "willingness_statements": [string],
                        "security_screenings": [string]
                        "behavioural_competencies": [string]
                    }}
                    
                    {format_instructions}
                    Use exact terminology from industry standards where applicable.""")
        ])

        # Chrome db for profiles data
        embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
        print('loading profiles vector store')
        self.profiles_vectorstore = Chroma(
            persist_directory=str(ROOT_DIR/'job_profiles_db2'),
            embedding_function=embeddings,
            collection_name="job_profiles"
        )
        print('done loading profiles vectorstore')

        # filter query construction

        metadata_field_info = [
            # AttributeInfo(
            #     name="title",
            #     description="Job title name",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="number",
            #     description="Unique job identification number",
            #     type="integer",
            # ),
            # AttributeInfo(
            #     name="type",
            #     description="Employment type classification",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="context",
            #     description="Organizational context of the position",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="views",
            #     description="Number of times this profile has been viewed",
            #     type="integer",
            # ),
            # AttributeInfo(
            #     name="role",
            #     description="Combined role ID and name in format 'ID - Name'",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="role_type",
            #     description="Combined role type ID and name in format 'ID - Name'", 
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="scopes",
            #     description="Comma-separated list of job responsibility areas",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="created_at",
            #     description="ISO 8601 timestamp of profile creation",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="updated_at",
            #     description="ISO 8601 timestamp of last profile update",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="row_index",
            #     description="Original CSV row index for reference",
            #     type="integer",
            # ),
            # New fields
            # AttributeInfo(
            #     name="classifications",
            #     description="Comma-separated list of classification names",
            #     type="string",
            # ),
            # AttributeInfo(
            #     name="organizations",
            #     description="Comma-separated list of organization/ministry names",
            #     type="string",
            # ),
            AttributeInfo(
                name="section",
                description="Document section header (e.g., 'Accountabilities', 'Security Screenings', 'Willingness Statements', 'Knowledge, Skills, and Abilities','Preferences','Professional Registration Requirements', 'Job Experience', 'Education', 'Behavioural Competencies')",
                type="string",
            )
        ]
        
        allowed_comparators = [
            "eq", "ne", "gt", "gte", "lt", "lte"  # Note 'contain' without 's'
        ]

        allowed_operators=["and", "or", "not"]

        # query_constructor_prompt=get_query_constructor_prompt(
        #     document_contents="Job profiles",
        #     attribute_info=metadata_field_info,
        #     # parser=parser,
        #     allowed_comparators=allowed_comparators,
        #     allowed_operators=allowed_operators,
        #     # examples=[
        #     #     # Add example with correct 'contain' usage
        #     #     (
        #     #         "Nurse roles in Band 1 classifications",
        #     #         """json
        #     #         {
        #     #             "query": "nurse band 1",
        #     #             "filter": "contain(\"classifications\", \"1 - Band 1\")"
        #     #         }"""
        #     #     )
        #     # ]
        # )

        # self.self_query_retriever = SelfQueryRetriever.from_llm(
        #     llm=self.llm,
        #     vectorstore=self.profiles_vectorstore,
        #     # (unclear) - high-level description of your document collection's content. 
        #     # It helps the LLM understand what semantic content to match when parsing user queries, while distinguishing it from metadata filters.
        #     document_contents="Job profile description and requirements",
        #     metadata_field_info=metadata_field_info,
        #     verbose=True,
        #     # This would actually allow users to say "What are two movies about dinosaurs" -> k==2
        #     # enable_limit=True,
        #     query_constructor_prompt=query_constructor_prompt,
        # )


        # NEW

        additional_instructions = """
        IMPORTANT FILTERING RULE:
        DO NOT ADD contains("content", "xxxx")) to the output, only fields specified below.
        """

        prompt = get_query_constructor_prompt(
            document_contents="Job profile"+additional_instructions,

            attribute_info=metadata_field_info,
            # allowed_operators=allowed_operators,
            allowed_comparators=allowed_comparators,
        )

        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | self.llm | output_parser

        # debug
        # ff = prompt.format(query="dummy question")
        
        # kk=query_constructor.invoke(
        #     {
        #         "query": "I need a profile for a nurse using accountabilities from band 1 classification terms"
        #     }
        # )
        
        # end debug

        self.self_query_retriever = SelfQueryRetriever(
            query_constructor=query_constructor,
            vectorstore=self.profiles_vectorstore,
            structured_query_translator=ChromaTranslator(),
        )

        # debug
        # documents = self.self_query_retriever.invoke("I need a profile for a nurse using accountabilities from Nurse 9 (C) classification terms", k=20)


        # def print_document_simple(doc):
        #     print("\n" + "="*80)
        #     # print(f"Document ID: {doc.id}")
        #     # print(f"Title: {doc.metadata['title']}")
        #     # print(f"Classification: {doc.metadata['classifications']}")
        #     # print(f"Organization: {doc.metadata['organizations']}")
        #     print(f"Section: {doc.metadata['section']}")
        #     print("-"*80)
        #     print(doc.page_content)
        #     print("="*80 + "\n")

        # for document in documents:
        #     print_document_simple(document)

        # filterDict={"section": "Accountabilities"}
        # filterDict={
        #     "classifications": {"$contains": "CLBC Clerk R9"}
        # }
        # I need a profile for a nurse using accountabilities from band 1 classification terms
        # context_docs2=self.profiles_vectorstore.similarity_search("nurse accountabilities band 1", k=3, filter=filterDict) #

        # alldocs=self.profiles_vectorstore.get()
        #end debug

        # END NEW

        self.chain = RunnablePassthrough.assign(
            context=lambda x: self.get_context(x["request"])
            ) | self.rag_prompt | self.llm | self.parser


    def count_tokens(self, text: str) -> int:
        return  len(self.encoder.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=text)
                ],
                model="mistral-small-latest"
            )
        ).tokens)

    def print_document_simple(self, doc):
        print("\n DOCUMENTS: " + "="*80)
        # print(f"Document ID: {doc.id}")
        # print(f"Title: {doc.metadata['title']}")
        # print(f"Classification: {doc.metadata['classifications']}")
        # print(f"Organization: {doc.metadata['organizations']}")
        # print(f"Section: {doc.metadata['section']}")
        print("-"*80)
        print(doc.page_content)
        print("="*80 + "\n")

    def get_context(self, query: str, k: int = 4):
        """Retrieve contextual documents from vector store"""
        context_docs = self.profiles_vectorstore.similarity_search('Accountabilities/Education/Job Experience/Professional Registration Requirements/Willingness Statements/Security Screenings'+query, k=10)

        # TODO = TO USE FILTERING
        # context_docs = self.self_query_retriever.invoke(query, k)

        # gg=self.profiles_vectorstore.get()
        

        total_tokens = 0
        processed_docs = []

        # Process documents one at a time, checking token count
        kk=0
        for document in context_docs:
            kk+=1
            self.print_document_simple(document)
            doc_tokens = self.count_tokens(document.page_content)
            
            # Check if adding this document would exceed token limit
            if total_tokens + doc_tokens <= 3500:
                processed_docs.append(document.page_content)
                total_tokens += doc_tokens
            else:
                print(f'TRUNCATING CONTEXT at {total_tokens} tokens k={kk}')
                break

        contextString = "\n\n".join([doc.page_content for doc in context_docs])
        tokenCount = self.count_tokens(contextString)

        print('context token length: ', tokenCount)

        return contextString

    def generate_profile(self, request, phase=2):
        # Phase 1: Basic generation
        generated = self.chain.invoke({
            "request": request,
            "format_instructions": self.parser.get_format_instructions()
        })

        print('generated from request: ', request)
        print(generated)
        
        if phase == 1:
            return generated
        
        # Phase 2: Semantic alignment with existing data
        aligned = {}
        for field, value in generated.items():
            if isinstance(value, list):
                aligned[field] = [
                    self._find_exact_match(item, field) or item
                    for item in value
                ]
            else:
                # existing = self._find_exact_match(value, field)
                # aligned[field] = existing or value
                aligned[field] = value
        
        return aligned

    
    # def _process_csv_with_pandas(self):
    #     """Process CSV with pandas for proper JSON handling"""
    #     docs = []
    #     seen_accountabilities = {}

    #     DATA_DIR = ROOT_DIR / "data"
    #     fpath = DATA_DIR / self.csv_path
        
    #     # Read CSV with pandas
    #     df = pd.read_csv(fpath)
        
    #     for idx, row in df.iterrows():
    #         try:
    #             # Process accountabilities with JSON parsing
    #             if pd.notna(row['accountabilities']):
    #                 try:
    #                     accountabilities = json.loads(row['accountabilities'])
    #                     for item in accountabilities:
    #                         text = item['text']
    #                         if text in seen_accountabilities:
    #                             # Update existing metadata
    #                             # seen_accountabilities[text].metadata['rows'].append(idx)
    #                             pass
    #                         else:
    #                             # Create new entry
    #                             doc = Document(
    #                                 page_content=f"{text}",
    #                                 metadata={
    #                                     "field": "accountabilities",
    #                                     "rows": idx,
    #                                     "sources": str(fpath)
    #                                 }
    #                             )
    #                             seen_accountabilities[text] = doc
    #                 except json.JSONDecodeError as e:
    #                     print(f"Row {idx}: Invalid JSON in accountabilities - {str(e)}")
                        
    #         except KeyError as e:
    #             print(f"Row {idx}: Missing column - {str(e)}")
                
    #     # Add aggregated documents
    #     docs.extend(seen_accountabilities.values())

    #     return docs


    def _process_csv_with_pandas(self):
        """Process CSV with pandas for proper JSON handling of multiple array fields"""
        docs = []
        seen_items = {field: {} for field in array_fields.keys()}

        DATA_DIR = ROOT_DIR / "data"
        fpath = DATA_DIR / self.csv_path
        
        # Read CSV with pandas
        df = pd.read_csv(fpath)
        
        for idx, row in df.iterrows():
            try:
                # Process each array field
                for field, display_name in array_fields.items():
                    if pd.notna(row[field]):
                        try:
                            items = json.loads(row[field])
                            for item in items:
                                if(field=="behavioural_competencies"):
                                    text = item['name'] + ": " + item['description']
                                else:
                                    text = item['text']
                                if text in seen_items[field]:
                                    # Update existing metadata
                                    # seen_items[field][text].metadata['rows'].append(idx)
                                    pass
                                else:
                                    # Create new entry
                                    doc = Document(
                                        page_content=f"{text}",
                                        metadata={
                                            "field": field,
                                            "rows": idx,
                                            "sources": str(fpath)
                                        }
                                    )
                                    seen_items[field][text] = doc
                        except json.JSONDecodeError as e:
                            print(f"Row {idx}: Invalid JSON in {field} - {str(e)}")
                            
            except KeyError as e:
                print(f"Row {idx}: Missing column - {str(e)}")
        
        # Add all aggregated documents
        for field_items in seen_items.values():
            docs.extend(field_items.values())

        return docs

    def _find_exact_match(self, text, field_type):
        """Find best matching existing phrase using vector similarity"""
        results = self.vector_store.similarity_search_with_score(
            query=text,
            k=1,
            filter={"field": field_type}
        )

        # g=self.vector_store.get()

        if results:
            print('input: ', text)
            print('match: ', results[0][0].page_content)
            print('threshold: ', results[0][1])


        if results and results[0][1] <= self.threshold:
            content = results[0][0].page_content
            # Extract just the value part after field prefix
            print('REPLACED: ', text)
            print('TO: ', content)
            print('==========')
            return content 
        return "\\* " + text

# TODO: UNCOMMENT
_GENERATOR = JobProfileGenerator("job profiles/2025-02-07_profiles.csv")

async def handle_generate_profile(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    global _GENERATOR
    generator = _GENERATOR

    # Phase 1 usage
    result_json = generator.generate_profile(
        query,
        phase=2
    )
    
    # Convert JSON to markdown
    def json_to_markdown(data):
        markdown = ""
        
        # Then process any remaining fields that aren't in array_fields
        remaining_fields = set(data.keys()) - set(array_fields.keys())
        for key in remaining_fields:
            value = data[key]
            if value:  # Only process if value is not empty
                markdown += f"## {key.replace('_', ' ').title()}\n"
                if isinstance(value, list):
                    for item in value:
                        markdown += f"- {item}\n"
                else:
                    markdown += f"{value}\n"

        # First process the known array_fields in the defined order
        for key, section_title in array_fields.items():
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    if(len(value) > 0):
                        markdown += f"## {section_title}\n"
                        for item in value:
                            markdown += f"- {item}\n"
                elif not isinstance(value, list) and value:
                    markdown += f"## {section_title}\n{value}\n"
        
        

        return markdown.strip()

    markdown_content = json_to_markdown(result_json)

    return {
            "id": "chatcmpl-" + os.urandom(4).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": markdown_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }