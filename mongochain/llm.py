"""Multi-provider LLM abstraction for mongochain."""

from typing import Optional, Generator


class LLMClient:
    """Unified interface for multiple LLM providers.
    
    Supports OpenAI, Azure OpenAI, Anthropic Claude, and Google Gemini with a consistent API.
    
    Attributes:
        provider: The LLM provider being used
        model: The specific model being used
    """
    
    PROVIDERS = {
        "openai": {"default_model": "gpt-4o-mini"},
        "azure_openai": {"default_model": None},  # Model/deployment specified by user
        "anthropic": {"default_model": "claude-3-haiku-20240307"},
        "google": {"default_model": "gemini-1.5-flash"},
        "grove": {"default_model": "gpt-4o-mini"},
    }

    GROVE_BASE_URL = "https://grove-gateway-prod.azure-api.net/grove-foundry-prod/openai/v1"
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-15-preview"
    ):
        """Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai", "azure_openai", "anthropic", or "google")
            api_key: API key for the provider
            model: Specific model to use (uses provider default if None).
                   For azure_openai, this is the deployment name (required).
            azure_endpoint: Azure OpenAI endpoint URL (required for azure_openai)
                           Example: "https://your-resource.openai.azure.com/"
            azure_api_version: Azure OpenAI API version (default: "2024-02-15-preview")
            
        Raises:
            ValueError: If provider is not supported or Azure config is missing
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Must be one of: {', '.join(self.PROVIDERS.keys())}"
            )
        
        # Validate Azure OpenAI requirements
        if provider == "azure_openai":
            if not azure_endpoint:
                raise ValueError(
                    "azure_endpoint is required for 'azure_openai' provider. "
                    "Example: 'https://your-resource.openai.azure.com/'"
                )
            if not model:
                raise ValueError(
                    "model (deployment name) is required for 'azure_openai' provider."
                )
        
        self.provider = provider
        self.model = model or self.PROVIDERS[provider]["default_model"]
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._azure_api_version = azure_api_version
        self._client = None
        
        # Initialize the appropriate client
        self._init_client()
    
    def _init_client(self):
        """Initialize the provider-specific client."""
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        
        elif self.provider == "azure_openai":
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._azure_api_version
            )
            
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model)

        elif self.provider == "grove":
            from openai import OpenAI
            # ponytail: Grove is OpenAI-compatible; reuse OpenAI SDK with base_url + api-key header
            self._client = OpenAI(
                api_key="unused",
                base_url=self.GROVE_BASE_URL,
                default_headers={"api-key": self._api_key},
            )
    
    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Send messages and return response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend
            
        Returns:
            The assistant's response text
        """
        if self.provider in ("openai", "azure_openai", "grove"):
            return self._chat_openai(messages, system_prompt)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, system_prompt)
        elif self.provider == "google":
            return self._chat_google(messages, system_prompt)
    
    def chat_stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Send messages and stream response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend
            
        Yields:
            Chunks of the assistant's response text
        """
        if self.provider in ("openai", "azure_openai", "grove"):
            yield from self._stream_openai(messages, system_prompt)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages, system_prompt)
        elif self.provider == "google":
            yield from self._stream_google(messages, system_prompt)
    
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None
    ) -> dict:
        """Send messages with tool definitions and handle tool calls.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: List of tool definitions in OpenAI function format
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Dict with either:
            - {"type": "text", "content": str} for regular responses
            - {"type": "tool_call", "name": str, "arguments": dict} for single tool call
            - {"type": "tool_calls", "calls": [{"name": str, "arguments": dict}, ...]} for multiple tool calls
        """
        if self.provider in ("openai", "azure_openai", "grove"):
            return self._chat_with_tools_openai(messages, tools, system_prompt)
        elif self.provider == "anthropic":
            return self._chat_with_tools_anthropic(messages, tools, system_prompt)
        elif self.provider == "google":
            return self._chat_with_tools_google(messages, tools, system_prompt)
    
    # ==================== OpenAI ====================
    
    def _chat_openai(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for OpenAI."""
        all_messages = []
        
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        
        all_messages.extend(messages)
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages
        )
        
        return response.choices[0].message.content
    
    def _stream_openai(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for OpenAI."""
        all_messages = []
        
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        
        all_messages.extend(messages)
        
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # ==================== Anthropic ====================
    
    def _chat_anthropic(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for Anthropic Claude."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages
        )
        
        return response.content[0].text
    
    def _stream_anthropic(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for Anthropic Claude."""
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    # ==================== Google ====================
    
    def _chat_google(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Handle chat for Google Gemini."""
        history = []
        
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = self._client.start_chat(history=history)
        
        last_message = messages[-1]["content"] if messages else ""
        if system_prompt and not history:
            last_message = f"{system_prompt}\n\n{last_message}"
        
        response = chat.send_message(last_message)
        
        return response.text
    
    def _stream_google(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream chat for Google Gemini."""
        history = []
        
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = self._client.start_chat(history=history)
        
        last_message = messages[-1]["content"] if messages else ""
        if system_prompt and not history:
            last_message = f"{system_prompt}\n\n{last_message}"
        
        response = chat.send_message(last_message, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    # ==================== Tool Calling ====================
    
    def _chat_with_tools_openai(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None
    ) -> dict:
        """Handle tool calling for OpenAI."""
        import json
        
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)
        
        # Convert tools to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for tool in tools
        ]
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None
        )
        
        message = response.choices[0].message
        
        # Check if the model wants to call tools
        if message.tool_calls:
            # If multiple tool calls, return them all
            if len(message.tool_calls) > 1:
                calls = [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in message.tool_calls
                ]
                return {"type": "tool_calls", "calls": calls}
            
            # Single tool call - maintain backward compatibility
            tool_call = message.tool_calls[0]
            return {
                "type": "tool_call",
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
        
        return {"type": "text", "content": message.content}
    
    def _chat_with_tools_anthropic(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None
    ) -> dict:
        """Handle tool calling for Anthropic Claude."""
        # Convert tools to Anthropic format
        anthropic_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            }
            for tool in tools
        ]
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=messages,
            tools=anthropic_tools if anthropic_tools else None
        )
        
        # Collect all tool use blocks
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        
        if tool_use_blocks:
            # If multiple tool calls, return them all
            if len(tool_use_blocks) > 1:
                calls = [
                    {
                        "name": block.name,
                        "arguments": block.input
                    }
                    for block in tool_use_blocks
                ]
                return {"type": "tool_calls", "calls": calls}
            
            # Single tool call - maintain backward compatibility
            block = tool_use_blocks[0]
            return {
                "type": "tool_call",
                "name": block.name,
                "arguments": block.input
            }
        
        # Return text content
        for block in response.content:
            if block.type == "text":
                return {"type": "text", "content": block.text}
        
        return {"type": "text", "content": ""}
    
    def _chat_with_tools_google(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None
    ) -> dict:
        """Handle tool calling for Google Gemini."""
        # Google Gemini tool calling requires different setup
        # For now, fall back to regular chat with tool info in prompt
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )
        
        enhanced_prompt = system_prompt or ""
        if tools:
            enhanced_prompt += f"\n\nYou have access to these tools:\n{tool_descriptions}\n"
            enhanced_prompt += "If you need to use tools, respond with: TOOL_CALL: tool_name(arg1=value1, arg2=value2)"
            enhanced_prompt += "\nYou can make multiple tool calls by putting each on its own line."
        
        response_text = self._chat_google(messages, enhanced_prompt)
        
        # Parse for tool calls
        if "TOOL_CALL:" in response_text:
            import re
            matches = re.findall(r'TOOL_CALL:\s*(\w+)\((.*?)\)', response_text)
            
            if matches:
                calls = []
                for tool_name, args_str in matches:
                    # Parse simple arg=value pairs
                    arguments = {}
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            arguments[key.strip()] = value.strip().strip('"\'')
                    calls.append({"name": tool_name, "arguments": arguments})
                
                # If multiple tool calls, return them all
                if len(calls) > 1:
                    return {"type": "tool_calls", "calls": calls}
                
                # Single tool call - maintain backward compatibility
                return {
                    "type": "tool_call",
                    "name": calls[0]["name"],
                    "arguments": calls[0]["arguments"]
                }
        
        return {"type": "text", "content": response_text}
