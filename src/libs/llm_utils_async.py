"""
LLM utilities for OpenAI API interactions.
Provides token counting, cost calculation, and API interaction utilities.

some references:
https://medium.com/@Aman-tech/how-make-async-calls-to-openais-api-cfc35f252ebd

"""

#%%
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import json
import re
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel,Field
from typing import List
import tiktoken
from openai import (
    AsyncOpenAI,
    APIError, 
    RateLimitError, 
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    BadRequestError
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
#from utils import load_json, logging, exception_handler
import logging 
import datetime 
now = datetime.datetime.now()
name = os.getlogin()
USER = name.upper()
file_path = f"log/{USER}/{datetime.date.today()}"
os.makedirs(file_path,exist_ok=True)
filename = f"{file_path}/Exp-{now.hour}:{now.minute}.log"
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=filename,
    filemode="w",
    format=fmt
    )


# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
TOKENIZER = tiktoken.encoding_for_model("gpt-4")

# Response format options
RESPONSE_FORMATS = {
    "json": {"type": "json_object"},
    "text": None
}

@dataclass
class ModelPricing:
    prompt: float
    completion: float

OAI_PRICE_DICT: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(0.01, 0.03),
    "gpt-4o-mini": ModelPricing(0.005, 0.015)
}

def tiktoken_len(text: str) -> int:
    """Calculate the number of tokens in a text string."""
    tokens = TOKENIZER.encode(text, disallowed_special=())
    return len(tokens)

def normalize_model_name(model_name: str) -> str:
    """Normalize various model name formats to their standard form."""
    if model_name.startswith(("gpt-4o-mini")):
        return "gpt-4o-mini"
    elif model_name.startswith("gpt-4o"):
        return "gpt-4o"
    else:
        return model_name

def get_oai_fees(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate OpenAI API fees based on model and token usage.
    
    Args:
        model_name: Name of the OpenAI model
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
    
    Returns:
        Cost in USD
    """
    try:
        model_name = normalize_model_name(model_name)
        pricing = OAI_PRICE_DICT[model_name]
        return (pricing.prompt * prompt_tokens + pricing.completion * completion_tokens) / 1000
    except KeyError:
        return -1

def retry_openai_api(
    wait_exponential_multiplier: int = 1,
    wait_exponential_max: int = 60,
    max_attempts: int = 3
):
    """Retry decorator for OpenAI API calls with exponential backoff."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_exponential_multiplier, max=wait_exponential_max),
        retry=(
            retry_if_exception_type(APIError) |
            retry_if_exception_type(RateLimitError) |
            retry_if_exception_type(APIConnectionError) |
            retry_if_exception_type(InternalServerError)
        )
    )
class AsyncBSAgent: # AsyncBaseAgent    
    """Async base agent class for OpenAI API interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[Union[str, type[BaseModel]]] = None
    ):
        """Initialize the AsyncBSAgent."""
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get('OPENAI_API_KEY', None),
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.message_history: List[Dict[str, str]] = []

    @retry_openai_api()
    async def _get_api_response(
        self,
        model: str,
        conv_history: List[Dict[str, str]],
        temperature: float,
        stream: bool = False,
        response_format: Optional[Union[str, type[BaseModel]]] = None
    ) -> Any:
        """Make an API call to OpenAI."""
        kwargs = {
            "model": model,
            "messages": conv_history,
            "temperature": temperature,
            "stream": stream
        }
        
        # Set response format if specified
        r_format = response_format or self.response_format
        if r_format:
            kwargs["response_format"] = r_format
            kwargs.pop("stream") ## parse doesn't support stream
            return await self.client.beta.chat.completions.parse(**kwargs)
        else:
            return await self.client.chat.completions.create(**kwargs)

    async def get_completion(
        self,
        prompt_template: Union[Dict[str, str], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        conv_history: List[Dict[str, str]] = None,
        return_cost: bool = False,
        verbose: bool = True,
        stream: bool = False,
        response_format: Optional[Union[str, type[BaseModel]]] = None
    ) -> Union[Any, Tuple[Any, float]]:
        """
        Get a completion from the API.
        
        Args:
            prompt_input: Either a dictionary containing 'System' and/or 'Human' prompts,
                or a list of OpenAI message format dicts
            model: Optional model override
            temperature: Optional temperature override
            conv_history: Optional conversation history
            return_cost: Whether to return the cost along with the response
            verbose: Whether to log the cost
            stream: Whether to stream the response
            response_format: Override default response format ('json' or None for text)
            
        Returns:
            API response or tuple of (response, cost) if return_cost is True
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        conv_history = list(conv_history or [])

        messages = []
        if isinstance(prompt_template, dict):
            # Handle prompt template format
            if "System" in prompt_template or "system" in prompt_template:
                system_content = prompt_template.get("System") or prompt_template.get("system")
                messages.append({"role": "system", "content": system_content})
            if "Human" in prompt_template or "user" in prompt_template:
                user_content = prompt_template.get("Human") or prompt_template.get("user")
                messages.append({"role": "user", "content": user_content})
                
            if not messages:
                raise ValueError("Prompt template must contain either 'System'/'system' or 'user'/'Human' message.")
        else:
            # Handle direct OpenAI message format
            messages = prompt_template

        conv_history.extend(messages)
        response = await self._get_api_response(
            model, 
            conv_history, 
            temperature, 
            stream, 
            response_format
        )

        if not stream:
            cost = get_oai_fees(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            if verbose:
                logging.info(f"Cost for this request: ${cost:.4f}")
            if return_cost:
                return response, cost

        return response

    async def get_response_content(self, **kwargs) -> str:
        """Get just the content from a completion response."""
        response = await self.get_completion(**kwargs)
        if kwargs.get("response_format") or self.response_format:
            return response.choices[0].message.parsed
        else:
            return response.choices[0].message.content

    @staticmethod
    async def extract_json_string(text: str) -> str:
        """Extract JSON string from between ```json``` markers."""
        match = re.search(r'```json\s+(.*?)\s+```', text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return text.replace("```json", "").replace("```", "").strip()

    async def parse_load_json_str(self, js: str) -> Dict:
        """Parse JSON string from response."""
        return json.loads(await self.extract_json_string(js))
    
    async def connection_test(self):
        """Test the connection to the API."""
        test_prompt = {
            "system": "You are a helpful assistant.",
            "user": "test connection"
        }
        res = await self.get_response_content(prompt_template=test_prompt)
        print(res)
        return res
    
async def async_unit_test_structured_output():
    """Async version of the structured output test.""" 
    # Example using Pydantic for structured output
    class Capital(BaseModel):
        city: str = Field(..., description="Name of the capital city")
        country: str = Field(..., description="Name of the country")
        population: int = Field(..., description="Population count")
        landmarks: List[str] = Field(default_factory=list, description="Notable landmarks")

    class CapitalsResponse(BaseModel):
        capitals: List[Capital] = Field(default_factory=list)
    
    structured_prompt = {
        "system": "You are a helpful assistant that provides structured data about European capitals.",
        "user": """List 2 European capitals with their details. 
        Return as JSON matching this structure:
        {
            "capitals": [
                {
                    "city": string,
                    "country": string,
                    "population": integer,
                    "landmarks": array of strings
                }
            ]
        }"""
    }
    agent = AsyncBSAgent()
    print("\nStructured output response:")
    response_parsed = await agent.get_response_content(prompt_template=structured_prompt, response_format=CapitalsResponse)
    # Now we can work with the strongly-typed data
    for capital in response_parsed.capitals:
        print(f"\n{capital.city}, {capital.country}")
        print(f"Population: {capital.population:,}")
        print("Notable landmarks:", ", ".join(capital.landmarks))

    # Test structured output
    print("\nPydantic string output response:")
    response_json = await agent.get_response_content(prompt_template=structured_prompt, response_format=None)
    response_json = agent.extract_json_string(response_json)
    capitals_data = CapitalsResponse.model_validate_json(response_json)
    for capital in capitals_data.capitals:
        print(f"\n{capital.city}, {capital.country}")
        print(f"Population: {capital.population:,}")
        print("Notable landmarks:", ", ".join(capital.landmarks))
        
async def test_basic_completion():
    """Test basic completion functionality."""
    agent = AsyncBSAgent()
    prompt = {
        "system": "You are a helpful assistant.",
        "user": "What is the capital of France?"
    }
    print("\nBasic completion test:")
    response = await agent.get_response_content(prompt_template=prompt)
    print(response)
    return response

async def test_conversation_history():
    """Test conversation with history."""
    agent = AsyncBSAgent()
    conv_history = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! How can I help you today?"}
    ]
    follow_up = {
        "Human": "What's my name?"
    }
    print("\nConversation with history test:")
    response = await agent.get_response_content(prompt_template=follow_up, conv_history=conv_history)
    print(response)
    return response

async def test_batch_processing():
    """Test parallel processing of multiple prompts."""
    agent = AsyncBSAgent()
    
    # Create multiple prompts for parallel processing
    prompts = [
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of France?"
        },
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of Germany?"
        },
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of Italy?"
        }
    ]
    
    print("\nBatch processing test:")
    tasks = [agent.get_response_content(prompt_template=prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}:")
        print(response)
    return responses 

async def test_token_counter():
    """Test the token counting functionality."""
    print("\nToken counter test:")
    test_text = 'a test sentence, macroeconomist'
    token_count = tiktoken_len(test_text)
    print(f"Text: '{test_text}'")
    print(f"Token count: {token_count}")
    return token_count

#%%
async def main():
    """Main function running all tests."""
    # Load OpenAI API key from .env file
    from dotenv import load_dotenv
    env_path = '../.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    # Run individual tests and collect results
    results = {
        "token_count": await test_token_counter(),
        "basic_completion": await test_basic_completion(),
        "conversation": await test_conversation_history(),
        "batch_processing": await test_batch_processing(),
        "structured_output": await async_unit_test_structured_output()
    }
    
    return results
#%%
if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()    
    #%%
    results = asyncio.run(main())
    print("\nAll test results:", results)
# 