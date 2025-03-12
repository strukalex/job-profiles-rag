import os
import logging
from fastapi import HTTPException
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from fastapi.responses import JSONResponse
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from langchain.schema import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables.config import RunnableConfig
from typing import Any

from .token_limiter import token_limiter

# Get logger for this module
logger = logging.getLogger('azure_client')

class AzureClientWrapper:
    """
    A wrapper for Azure API client that incorporates token limiting.
    """
    def __init__(self):
        """Initialize the Azure client wrapper."""
        self.endpoint = os.getenv('AZURE_ENDPOINT')
        self.api_key = os.getenv('AZURE_API_KEY')
        
        logger.info(f"Initializing AzureClientWrapper with endpoint: {self.endpoint}")
        
        # Initialize the Azure client
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            model="Mistral-small"
        )
        
        # Initialize the tokenizer for counting tokens
        self.encoder = MistralTokenizer.from_model("mistral-small", strict=True)
        logger.info("Initialized MistralTokenizer for token counting")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text (str): The text to count tokens for
            
        Returns:
            int: The number of tokens
        """
        try:
            # Truncate text for logging to avoid huge log files
            log_text = text[:100] + "..." if len(text) > 100 else text
            logger.debug(f"Counting tokens for text: {log_text}")
            
            tokens = len(self.encoder.encode_chat_completion(
                ChatCompletionRequest(
                    messages=[UserMessage(content=text)],
                    model="mistral-small-latest"
                )
            ).tokens)
            
            logger.info(f"Token count: {tokens} for text length: {len(text)}")
            return tokens
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Return an estimate as fallback
            estimate = len(text.split()) * 1.3
            logger.warning(f"Using estimated token count: {estimate}")
            return int(estimate)
    
    def complete(self, messages, max_tokens=300, temperature=0.7):
        """
        Complete a chat conversation with token limiting.
        
        Args:
            messages (list): List of message dictionaries
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for generation
            
        Returns:
            object: The completion response
            
        Raises:
            HTTPException: If token limit is exceeded
        """
        # Count tokens in the prompt
        prompt_text = " ".join([msg["content"] for msg in messages])
        prompt_tokens = self.count_tokens(prompt_text)
        
        # Estimate total tokens (prompt + max_tokens for completion)
        estimated_total_tokens = prompt_tokens + max_tokens
        logger.info(f"API call preparation: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, estimated_total={estimated_total_tokens}")
        
        # Check if we have enough tokens left for today
        if not token_limiter.check_limit():
            logger.warning("Token limit exceeded, rejecting API call")
            return JSONResponse(
                        content={
                            "error": {
                                "message": f"Daily token limit would be exceeded. Please try again tomorrow.",
                                "type": "rate_limit_error",
                                "param": None,
                                "code": "rate_limit_exceeded"
                            }
                        },
                        status_code=429,
                    )
        
        # Make the API call
        logger.info(f"Making Azure API call with temperature={temperature}, max_tokens={max_tokens}")
        try:
            response = self.client.complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Count actual tokens used in the response
            if response and response.choices:
                completion_text = response.choices[0].message.content
                completion_tokens = self.count_tokens(completion_text)
                actual_total_tokens = prompt_tokens + completion_tokens
                
                logger.info(f"API call successful: completion_tokens={completion_tokens}, actual_total={actual_total_tokens}")
                
                # Add the actual tokens used to our counter
                token_added = token_limiter.add_tokens(actual_total_tokens)
                logger.info(f"Tokens added to counter: {actual_total_tokens}, success={token_added}")
            else:
                # If no response, just count the prompt tokens
                logger.warning("API call returned no choices, counting only prompt tokens")
                token_added = token_limiter.add_tokens(prompt_tokens)
                logger.info(f"Prompt tokens added to counter: {prompt_tokens}, success={token_added}")
            
            return response
        except Exception as e:
            logger.error(f"Error in Azure API call: {e}")
            raise

# Create a singleton instance
azure_client = AzureClientWrapper()
logger.info("AzureClientWrapper singleton instance created")

def get_langchain_azure_model(model_name="Mistral-small", api_version="2024-05-01-preview", temperature=0.7, top_p=None, **kwargs):
    """
    Get a LangChain Azure AI Chat Completions Model with token limiting.
    
    Args:
        model_name (str): The model name
        api_version (str): The API version
        temperature (float, optional): Temperature for generation. Defaults to 0.7.
        top_p (float, optional): Top p for nucleus sampling. Defaults to None.
        **kwargs: Additional model parameters
        
    Returns:
        AzureAIChatCompletionsModel: The LangChain model
    """
    logger.info(f"Creating LangChain Azure model: model={model_name}, temperature={temperature}, top_p={top_p}")
    
    # Extract model_kwargs from kwargs
    model_kwargs = kwargs.get('model_kwargs', {})
    logger.debug(f"Model kwargs: {model_kwargs}")
    
    # Create the LangChain model
    try:
        # Create a subclass that overrides the necessary method
        class TokenTrackedModel(AzureAIChatCompletionsModel):
            async def ainvoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, *, stop: list[str] | None = None, **kwargs: Any) -> BaseMessage:
                logger.info("LangChain model ainvoke called")
                
                # Get the prompt tokens from input
                try:
                    if isinstance(input, list):
                        prompt_text = " ".join([msg.content for msg in input])
                    else:
                        prompt_text = str(input)
                    prompt_tokens = azure_client.count_tokens(prompt_text)
                    logger.info(f"LangChain call prompt tokens: {prompt_tokens}")
                except Exception as e:
                    logger.error(f"Error extracting content from input: {e}")
                    prompt_tokens = 0
                
                # Get current token usage stats
                stats = token_limiter.get_usage_stats()
                remaining_tokens = stats["remaining_tokens"]
                
                # Estimate total tokens needed (prompt + conservative completion estimate)
                estimated_completion_tokens = 500  # Conservative estimate for completion
                estimated_total = prompt_tokens + estimated_completion_tokens
                
                # Check if we have enough tokens remaining BEFORE making the request
                if estimated_total > remaining_tokens:
                    logger.warning(f"Token limit would be exceeded: need ~{estimated_total} tokens but only {remaining_tokens} remaining")
                    return JSONResponse(
                        content={
                            "error": {
                                "message": f"Daily token limit would be exceeded. Please try again tomorrow.",
                                "type": "rate_limit_error",
                                "param": None,
                                "code": "rate_limit_exceeded"
                            }
                        },
                        status_code=429,
                    )
                
                # Make the API call
                try:
                    logger.info("Making LangChain API call")
                    response = await super().ainvoke(input, config=config, stop=stop, **kwargs)
                    logger.info("LangChain API call successful")
                    
                    # Get token usage from response and add to counter
                    try:
                        completion_text = response.content
                        completion_tokens = azure_client.count_tokens(completion_text)
                        total_tokens = prompt_tokens + completion_tokens
                        
                        # Just add the tokens without checking - we already made the request
                        token_limiter.add_tokens(total_tokens)
                        logger.info(f"Added {total_tokens} tokens to counter after successful completion")
                        
                    except Exception as e:
                        logger.error(f"Error counting completion tokens: {e}")
                        # Even for errors, add conservative estimate since we made the request
                        conservative_total = prompt_tokens + estimated_completion_tokens
                        token_limiter.add_tokens(conservative_total)
                        logger.warning(f"Using estimated total tokens: {conservative_total}")
                    
                    return response
                except Exception as e:
                    logger.error(f"Error in LangChain API call: {e}")
                    raise

        # Create an instance of our subclass instead
        model = TokenTrackedModel(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credential=os.getenv('AZURE_API_KEY'),
            model_name=model_name,
            api_version=api_version,
            temperature=temperature,
            top_p=top_p,
            model_kwargs=model_kwargs
        )
        logger.info("LangChain Azure model created successfully")
    except Exception as e:
        logger.error(f"Error creating LangChain Azure model: {e}")
        raise
    
    return model 