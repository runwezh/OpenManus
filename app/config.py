import threading
import tomllib
from pathlib import Path
from typing import Dict
import os
from dotenv import load_dotenv  # 需要安装: pip install python-dotenv

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        # 加载 .env 文件
        dotenv_path = Path(PROJECT_ROOT) / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
        
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        # 添加这行来获取 provider
        provider = base_llm.get("provider", "openai")

        default_settings = {
            "provider": provider,  # 添加 provider 参数
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url", "http://default-url.com"),
            "api_key": base_llm.get("api_key", "default-api-key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
        }

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            }
        }

        # 从环境变量覆盖 API 密钥
        if "DEEPSEEK_API_KEY" in os.environ and "deepseek" in llm_overrides:
            llm_overrides["deepseek"]["api_key"] = os.environ["DEEPSEEK_API_KEY"]
        
        if "OPENAI_API_KEY" in os.environ and "openai" in llm_overrides:
            llm_overrides["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]

        self._config = AppConfig(**config_dict)

        # Ensure the config file has valid values
        if not self._config.llm["default"].base_url or not self._config.llm["default"].api_key:
            raise ValueError("Invalid configuration: 'base_url' and 'api_key' must be set for 'llm.default'")
        if "ollama" in self._config.llm and not self._config.llm["ollama"].api_key:
            raise ValueError("Invalid configuration: 'api_key' must be set for 'llm.ollama'")
        if "vision" in self._config.llm and (not self._config.llm["vision"].base_url or not self._config.llm["vision"].api_key):
            raise ValueError("Invalid configuration: 'base_url' and 'api_key' must be set for 'llm.vision'")

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm


config = Config()
