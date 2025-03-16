import os
import json
import logging
import re
from typing import Dict, List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field
from anthropic import Anthropic

class ClaudeAgent:
    """Base agent class for Anthropic Claude API interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet",
        temperature: float = 0.0,
        max_tokens: int = 4000
    ):
        """
        Initialize the ClaudeAgent.
        
        Args:
            api_key: Anthropic API key. If None, uses environment variable
            model: Claude model identifier to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate in response
        """
        self.client = Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message_history: List[Dict[str, str]] = []
    
    def _get_api_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make an API call to Anthropic Claude."""
        try:
            # Convert from OpenAI format to Claude format
            claude_messages = []
            system_content = None
            
            # Extract system message if present
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    claude_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = self.client.messages.create(
                model=model,
                messages=claude_messages,
                system=system_content,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            raise
    
    def get_completion(
        self,
        prompt_template: Union[Dict[str, str], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        conv_history: List[Dict[str, str]] = None,
        return_cost: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Union[Any, Tuple[Any, float]]:
        """
        Get a completion from the Claude API.
        
        Args:
            prompt_template: Either a dictionary containing 'system' and/or 'user' prompts,
                or a list of message format dicts
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            conv_history: Optional conversation history
            return_cost: Whether to return the cost along with the response
            verbose: Whether to log the cost
            
        Returns:
            API response or tuple of (response, cost) if return_cost is True
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        conv_history = list(conv_history or [])
        
        # Convert dict prompt to list format if needed
        if isinstance(prompt_template, dict):
            messages = []
            if "system" in prompt_template:
                messages.append({"role": "system", "content": prompt_template["system"]})
            if "user" in prompt_template:
                messages.append({"role": "user", "content": prompt_template["user"]})
            if "assistant" in prompt_template:
                messages.append({"role": "assistant", "content": prompt_template["assistant"]})
        else:
            messages = prompt_template

        conv_history.extend(messages)
        response = self._get_api_response(
            model, 
            conv_history, 
            temperature, 
            max_tokens,
            **kwargs
        )

        # Claude doesn't provide token usage in the same way as OpenAI
        if verbose:
            logging.info(f"Response received from Claude model: {model}")
        
        return response
    
    def get_response_content(self, **kwargs) -> str:
        """Get just the content from a completion response."""
        response = self.get_completion(**kwargs)
        return response.content[0].text
    
    @staticmethod
    def extract_json_string(text: str) -> str:
        """Extract JSON string from between ```json``` markers."""
        match = re.search(r'```json\s+(.*?)\s+```', text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return text.replace("```json", "").replace("```", "").strip()

    def parse_load_json_str(self, js: str) -> Dict:
        """Parse JSON string from response."""
        return json.loads(self.extract_json_string(js))
    
    def connection_test(self, user_prompt=None):
        """Test the connection to the API."""
        if user_prompt is None:
            user_prompt = "What is the capital of France?"
        test_prompt = {
            "system": "You are a helpful assistant.",
            "user": user_prompt
        }
        res = self.get_response_content(prompt_template=test_prompt)
        print(res)
