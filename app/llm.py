from typing import Dict, List, Literal, Optional, Union

from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import Message
from .ollama_client import OllamaClient
from .llm_client import create_llm_client


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name=None, llm_config=None):
        if llm_config is None:
            llm_config = {}
        
        # 获取配置中的 provider
        provider = config.llm["default"].provider if hasattr(config.llm["default"], "provider") else "deepseek"
        
        # 基本模型配置来自 default 或被传入的 llm_config 覆盖
        self.model = llm_config.get("model") or config.llm["default"].model
        self.max_tokens = llm_config.get("max_tokens") or config.llm["default"].max_tokens
        self.temperature = llm_config.get("temperature") or config.llm["default"].temperature
        
        # 根据 provider 获取特定配置
        if provider in config.llm:
            provider_config = config.llm[provider]
            # 使用特定提供商配置，如果有的话
            self.base_url = provider_config.base_url
            self.api_key = provider_config.api_key if provider != "ollama" else None
        else:
            # 回退到默认配置
            self.base_url = config.llm["default"].base_url
            self.api_key = config.llm["default"].api_key if provider != "ollama" else None
            
        # Ollama 特殊处理
        if provider == "ollama" and not self.base_url:
            self.base_url = "http://localhost:11434"
        
        # 导入并创建适当的 LLM 客户端
        from .llm_client import create_llm_client
        self.llm_client = create_llm_client(
            provider=provider,
            model=self.model,
            config={
                "base_url": self.base_url,
                "api_key": self.api_key
            }
        )
        
        # 打印初始化信息以便调试
        logger.info(f"LLM 初始化完成 - 提供者: {provider}, 模型: {self.model}, URL: {self.base_url}")

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                return response.choices[0].message.content

            # Streaming request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.
        
        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments
            
        Returns:
            ChatCompletionMessage: The model's response
            
        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            Exception: For API errors or unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")
                
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)
                
            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")
                        
            # 设置温度
            temp = temperature if temperature is not None else self.temperature
            
            # 使用我们的 LLM 客户端
            response = await self.llm_client.chat_completions(
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            # 处理响应
            return response
            
        except ValueError as ve:
            logger.error(f"Value error in ask_tool: {str(ve)}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {str(e)}")
            raise
