import logging
import re
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# 在文件顶部加载 .env
load_dotenv()

logger = logging.getLogger(__name__)

# 通用对象类用于模拟 OpenAI 响应结构
class ToolCallObject:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = function

class FunctionObject:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class MessageObject:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

class ChoiceObject:
    def __init__(self, message, finish_reason="stop", index=0):
        self.message = message
        self.finish_reason = finish_reason
        self.index = index

class UsageObject:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

class CompletionObject:
    def __init__(self, id, object_type, created, model, choices, usage):
        self.id = id
        self.object = object_type
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.tool_calls = None  # 顶层 tool_calls 属性
        
        # 添加直接访问内容的属性
        self.content = choices[0].message.content if choices and hasattr(choices[0], 'message') else ""

# 基础 LLM 客户端类
class BaseLLMClient:
    def __init__(self, model: str, **kwargs):
        self.model = model
        
    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """从内容中提取工具调用的通用方法"""
        tool_calls = []
        
        # 查找 <tool>function_name({"param": "value"})</tool> 格式的函数调用
        tool_pattern = re.compile(r'<tool>(.*?)\((.*?)\)</tool>', re.DOTALL)
        matches = tool_pattern.findall(content)
        
        # 同时查找可能表示函数调用的 JSON 对象
        json_pattern = re.compile(r'\{[\s\S]*?"function"[\s\S]*?:[\s\S]*?\{[\s\S]*?"name"[\s\S]*?:[\s\S]*?"(.*?)"[\s\S]*?"arguments"[\s\S]*?:[\s\S]*?([\s\S]*?)\}[\s\S]*?\}')
        json_matches = json_pattern.findall(content)
        
        # 处理正则表达式匹配
        for i, (func_name, args_str) in enumerate(matches):
            try:
                args_str = args_str.strip()
                if args_str.startswith('{') and args_str.endswith('}'):
                    args = args_str
                else:
                    args = '{}'
                    
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": func_name.strip(),
                        "arguments": args
                    }
                })
            except Exception as e:
                logger.warning(f"解析工具调用失败: {e}")
        
        # 处理 JSON 匹配
        for i, (func_name, args_str) in enumerate(json_matches):
            try:
                tool_calls.append({
                    "id": f"json_call_{i}",
                    "type": "function",
                    "function": {
                        "name": func_name.strip(),
                        "arguments": args_str.strip()
                    }
                })
            except Exception as e:
                logger.warning(f"解析 JSON 工具调用失败: {e}")
        
        return tool_calls
    
    def _create_completion_object(self, id_prefix: str, content: str, model: str) -> CompletionObject:
        """创建标准化的完成对象"""
        parsed_tool_calls = self._parse_tool_calls(content)
        
        # 创建工具调用对象
        tool_call_objects = []
        for tc in parsed_tool_calls:
            function_obj = FunctionObject(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
            )
            tool_call_obj = ToolCallObject(
                id=tc["id"],
                type=tc["type"],
                function=function_obj
            )
            tool_call_objects.append(tool_call_obj)
        
        # 创建消息对象
        message_obj = MessageObject(
            role="assistant", 
            content=content,
            tool_calls=tool_call_objects if tool_call_objects else None
        )
        
        # 创建选择对象
        choice_obj = ChoiceObject(message=message_obj)
        
        # 创建使用情况对象
        usage_obj = UsageObject()
        
        # 创建最终的响应对象
        response_obj = CompletionObject(
            id=f"chatcmpl-{id_prefix}",
            object_type="chat.completion",
            created=1679351277,
            model=model,
            choices=[choice_obj],
            usage=usage_obj
        )
        
        # 添加顶级工具调用
        if tool_call_objects:
            response_obj.tool_calls = tool_call_objects
            
        return response_obj

    async def chat_completions(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.0, 
                              max_tokens: Optional[int] = None) -> Any:
        """必须在子类中实现的聊天完成方法"""
        raise NotImplementedError("必须在子类中实现此方法")

# Ollama 客户端实现
class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "deepseek-r1:32b", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model)
        self.base_url = base_url
        logger.info(f"初始化 OllamaClient，模型: {model}，URL: {base_url}")

    async def chat_completions(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.0, 
                              max_tokens: Optional[int] = None) -> Any:
        """调用 Ollama API 并格式化类 OpenAI 响应"""
        
        # 清理模型名称（移除提供者前缀）
        model = self.model.split('/')[-1] if '/' in self.model else self.model
        
        # 准备请求体
        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        # 添加 max_tokens（如果提供）
        if max_tokens:
            request_body["options"]["num_predict"] = max_tokens
        
        try:
            logger.debug(f"发送请求到 Ollama: {request_body}")
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/chat", json=request_body) as response:
                    if response.status == 200:
                        ollama_response = await response.json()
                        logger.debug("从 Ollama 接收到成功响应")
                        
                        # 从 Ollama 响应中获取内容
                        content = ollama_response.get("message", {}).get("content", "")
                        
                        # 创建标准化的完成对象
                        return self._create_completion_object("ollama", content, model)
                    else:
                        text = await response.text()
                        logger.error(f"Ollama API 错误: {response.status} - {text}")
                        raise Exception(f"Ollama API 错误: {response.status} - {text}")
        
        except Exception as e:
            logger.exception(f"调用 Ollama 错误: {str(e)}")
            raise

# DeepSeek 客户端实现
class DeepSeekClient(BaseLLMClient):
    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", api_key: str = "", **kwargs):
        super().__init__(model)
        self.base_url = base_url
        self.api_key = api_key
        # 添加调试日志
        logger.info(f"初始化 DeepSeekClient，模型: {model}，URL: {base_url}")
        # 安全地检查 API 密钥是否存在
        if not api_key:
            logger.warning("⚠️ DeepSeek API 密钥为空！请检查 .env 文件或配置")
        else:
            # 不打印完整密钥，只显示前几个字符
            masked_key = api_key[:6] + "..." if len(api_key) > 6 else "***"
            logger.info(f"使用 DeepSeek API 密钥: {masked_key}")

    async def chat_completions(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.0, 
                              max_tokens: Optional[int] = None) -> Any:
        """调用 DeepSeek API 并格式化类 OpenAI 响应"""
        
        # 清理模型名称（移除提供者前缀）
        model = self.model.split('/')[-1] if '/' in self.model else self.model
        
        # 准备请求体
        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature
        }
        
        # 添加 max_tokens（如果提供）
        if max_tokens:
            request_body["max_tokens"] = max_tokens
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.debug(f"发送请求到 DeepSeek: {request_body}")
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", json=request_body, headers=headers) as response:
                    if response.status == 200:
                        deepseek_response = await response.json()
                        logger.debug("从 DeepSeek 接收到成功响应")
                        
                        # 从 DeepSeek 响应中获取内容
                        content = deepseek_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # 创建标准化的完成对象
                        return self._create_completion_object("deepseek", content, model)
                    else:
                        text = await response.text()
                        logger.error(f"DeepSeek API 错误: {response.status} - {text}")
                        raise Exception(f"DeepSeek API 错误: {response.status} - {text}")
        
        except Exception as e:
            logger.exception(f"调用 DeepSeek 错误: {str(e)}")
            raise

# OpenAI 客户端实现
class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-3.5-turbo", base_url: str = "https://api.openai.com/v1", api_key: str = "", **kwargs):
        super().__init__(model)
        self.base_url = base_url
        self.api_key = api_key
        logger.info(f"初始化 OpenAIClient，模型: {model}，URL: {base_url}")

    async def chat_completions(self, messages: List[Dict[str, str]], 
                              temperature: float = 0.0, 
                              max_tokens: Optional[int] = None) -> Any:
        """直接调用 OpenAI API 并返回结果"""
        
        # 准备请求体
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        # 添加 max_tokens（如果提供）
        if max_tokens:
            request_body["max_tokens"] = max_tokens
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            logger.debug(f"发送请求到 OpenAI: {request_body}")
            response = requests.post(f"{self.base_url}/chat/completions", json=request_body, headers=headers)
            
            if response.status_code == 200:
                openai_response = response.json()
                logger.debug("从 OpenAI 接收到成功响应")
                
                # 从 OpenAI 响应中获取内容
                content = openai_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 创建标准化的完成对象（或者可以直接返回原始 OpenAI 响应）
                return self._create_completion_object("openai", content, self.model)
            else:
                logger.error(f"OpenAI API 错误: {response.status_code} - {response.text}")
                raise Exception(f"OpenAI API 错误: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.exception(f"调用 OpenAI 错误: {str(e)}")
            raise

# 客户端工厂方法
def create_llm_client(provider: str, model: str, config: dict) -> BaseLLMClient:
    """根据提供者和配置创建适当的 LLM 客户端"""
    
    if provider == "ollama":
        base_url = config.get("base_url", "http://localhost:11434")
        return OllamaClient(model=model, base_url=base_url)
    
    elif provider == "deepseek":
        base_url = config.get("base_url", "https://api.deepseek.com")
        # 优先使用环境变量中的 API 密钥
        api_key = os.environ.get("DEEPSEEK_API_KEY", "") or config.get("api_key", "")
        return DeepSeekClient(model=model, base_url=base_url, api_key=api_key)
    
    elif provider == "openai":
        base_url = config.get("base_url", "https://api.openai.com/v1")
        # 优先使用环境变量中的 API 密钥
        api_key = os.environ.get("OPENAI_API_KEY", "") or config.get("api_key", "")
        return OpenAIClient(model=model, base_url=base_url, api_key=api_key)
    
    else:
        raise ValueError(f"不支持的 LLM 提供者: {provider}")