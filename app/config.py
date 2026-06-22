import threading
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Dict, List, Optional

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
    api_type: str = Field(..., description="AzureOpenai or Openai")
    api_version: str = Field(..., description="Azure Openai version if AzureOpenai")

    # ---- Optional: force the OpenAI token-budget parameter name --------------------
    # None  -> auto-detect by model name (o1/o3/o4/gpt-5 -> max_completion_tokens)
    # True  -> always send max_completion_tokens (and omit temperature)
    # False -> always send max_tokens
    use_max_completion_tokens: Optional[bool] = Field(
        default=None,
        description="Force max_completion_tokens vs max_tokens; None = auto-detect",
    )

    # ---- Optional local-HuggingFace backend fields (api_type == "hf_local") -------
    device_map: Optional[str] = Field(
        default=None, description="HF device_map, e.g. 'auto'"
    )
    torch_dtype: Optional[str] = Field(
        default=None, description="torch dtype: bfloat16 | float16 | float32 | auto"
    )
    load_in_4bit: bool = Field(default=False, description="Load model in 4-bit (bitsandbytes)")
    load_in_8bit: bool = Field(default=False, description="Load model in 8-bit (bitsandbytes)")
    attn_implementation: Optional[str] = Field(
        default=None, description="HF attn implementation, e.g. 'eager'"
    )
    enable_thinking: bool = Field(
        default=False, description="Enable Qwen <think> chat-template mode"
    )

    # ---- Optional profiling fields ------------------------------------------------
    profile_cache: bool = Field(
        default=False, description="Collect KV/V-cache summaries for this LLM"
    )
    profile_dir: Optional[str] = Field(
        default="record", description="Output directory for profiling artifacts"
    )


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )
    search_config: Optional[SearchSettings] = Field(
        None, description="Search configuration"
    )

    class Config:
        arbitrary_types_allowed = True


# Optional keys that may appear in a per-model [llm.xxx] override and should be
# carried through to LLMSettings. Defaults are handled by the model itself.
_OPTIONAL_LLM_KEYS = (
    "device_map",
    "torch_dtype",
    "load_in_4bit",
    "load_in_8bit",
    "attn_implementation",
    "enable_thinking",
    "profile_cache",
    "profile_dir",
    "use_max_completion_tokens",
)


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
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # Carry through any optional local-HF / profiling keys present at the top level.
        for key in _OPTIONAL_LLM_KEYS:
            if key in base_llm:
                default_settings[key] = base_llm[key]

        # handle browser config.
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # handle proxy settings.
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # filter valid browser config parameters.
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # if there is proxy settings, add it to the parameters.
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # only create BrowserSettings when there are valid parameters.
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "browser_config": browser_settings,
            "search_config": search_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._config.search_config


config = Config()
