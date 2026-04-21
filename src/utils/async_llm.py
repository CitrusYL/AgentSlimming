import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from openai import AsyncOpenAI

from src.core.formatter import BaseFormatter, FormatError
from src.utils.cost_tracker import LLMCostTracker
from src.utils.logs import logger


DEFAULT_CONFIG_NAME = "config.yaml"
CONFIG_ENV_VAR = "AGENT_SLIMMING_CONFIG"
DEFAULT_TOP_P = 1.0


class LLMConfig:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, **overrides: Any):
        data = dict(config or {})
        data.update(overrides)

        self.model = _required_str_config(data, "model")
        self.temperature = _required_float_config(data, "temperature")
        self.key = _required_str_config(data, "api_key", aliases=("key",))
        self.base_url = _required_str_config(data, "base_url")
        self.top_p = _float_config_with_default(data, "top_p", DEFAULT_TOP_P)
        self.api_type = _required_str_config(data, "api_type")
        self.input_price = _optional_float_config(data.get("input_price"))
        self.output_price = _optional_float_config(data.get("output_price"))

    @property
    def api_key(self) -> Optional[str]:
        return self.key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "key": self.key,
            "base_url": self.base_url,
            "top_p": self.top_p,
            "api_type": self.api_type,
            "input_price": self.input_price,
            "output_price": self.output_price,
        }

    def require_pricing(self) -> tuple[float, float]:
        if self.input_price is None or self.output_price is None:
            raise ValueError(
                f"Missing pricing for model {self.model}. "
                "Set input_price and output_price in config/config.yaml."
            )
        return self.input_price, self.output_price


class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""

    _default_configs: Dict[Path, "LLMsConfig"] = {}
    _active_config_path: Optional[Path] = None

    def __init__(
        self,
        config_dict: Optional[Mapping[str, Mapping[str, Any]]] = None,
        source_path: Optional[Path] = None,
    ):
        """Initialize with an optional configuration dictionary"""
        self.configs = dict(config_dict or {})
        self.source_path = source_path

    @classmethod
    def default(cls, config_path: Optional[str | Path] = None):
        """Get or create a default configuration from YAML file"""
        if config_path is None and cls._active_config_path is not None:
            resolved_path = cls._active_config_path
        else:
            resolved_path = cls.resolve_config_path(config_path)

        if resolved_path not in cls._default_configs:
            cls._default_configs[resolved_path] = cls.from_file(resolved_path)
        cls._active_config_path = resolved_path
        return cls._default_configs[resolved_path]

    @classmethod
    def from_file(cls, config_path: str | Path) -> "LLMsConfig":
        resolved_path = Path(config_path).expanduser().resolve()
        with resolved_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        if not isinstance(config_data, dict):
            raise ValueError(f"LLM config must be a mapping: {resolved_path}")

        model_configs = config_data.get("models", config_data)
        if not isinstance(model_configs, dict):
            raise ValueError(f"LLM config 'models' must be a mapping: {resolved_path}")

        return cls(model_configs, source_path=resolved_path)

    @classmethod
    def resolve_config_path(cls, config_path: Optional[str | Path] = None) -> Path:
        candidates = cls._config_path_candidates(config_path)
        checked_paths = []

        for path in candidates:
            resolved_path = cls._resolve_path(path)
            checked_paths.append(str(resolved_path))
            if resolved_path.exists():
                return resolved_path

        raise FileNotFoundError(
            "No LLM configuration file found. "
            f"Set {CONFIG_ENV_VAR} or create one of: {', '.join(checked_paths)}"
        )

    @classmethod
    def _config_path_candidates(cls, config_path: Optional[str | Path]) -> list[Path]:
        if config_path is not None:
            return [Path(config_path)]

        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            return [Path(env_path)]

        repo_root = Path(__file__).resolve().parents[2]
        return [
            repo_root / "config" / DEFAULT_CONFIG_NAME,
            Path.cwd() / "config" / DEFAULT_CONFIG_NAME,
            Path.cwd() / DEFAULT_CONFIG_NAME,
        ]

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        path = path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            available = ", ".join(self.get_all_names()) or "<empty>"
            raise ValueError(
                f"Configuration for {llm_name} not found. Available models: {available}"
            )

        config = self.configs[llm_name]
        if not isinstance(config, Mapping):
            raise ValueError(f"Configuration for {llm_name} must be a mapping")

        return LLMConfig(config, model=config.get("model", llm_name))

    def add_config(self, name: str, config: Mapping[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = dict(config)

    def get_all_names(self) -> list[str]:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())


def _required_present_config(
    data: Mapping[str, Any],
    key: str,
    aliases: tuple[str, ...] = (),
) -> Any:
    for candidate in (key, *aliases):
        if candidate in data:
            return data[candidate]
    names = ", ".join((key, *aliases))
    raise ValueError(f"Missing required LLM config field: {names}")


def _required_str_config(data: Mapping[str, Any], key: str, aliases: tuple[str, ...] = ()) -> str:
    value = _required_present_config(data, key, aliases)
    if value is None:
        raise ValueError(f"Missing required LLM config field: {key}")
    return str(value)


def _required_float_config(data: Mapping[str, Any], key: str) -> float:
    value = _required_present_config(data, key)
    if value is None:
        raise ValueError(f"Missing required LLM config field: {key}")
    return float(value)


def _float_config_with_default(data: Mapping[str, Any], key: str, default: float) -> float:
    if key not in data or data[key] is None:
        return default
    return float(data[key])


def _optional_float_config(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def resolve_llm_config(llm_config: Any) -> LLMConfig:
    if isinstance(llm_config, LLMConfig):
        return llm_config

    if isinstance(llm_config, str):
        return LLMsConfig.default().get(llm_config)

    if isinstance(llm_config, Mapping):
        config_data = dict(llm_config)
        model_name = config_data.get("model")
        if model_name:
            base_config = LLMsConfig.default().get(str(model_name)).to_dict()
            base_config.update({
                key: value for key, value in config_data.items() if value is not None
            })
            config_data = base_config
        return LLMConfig(config_data)

    raise TypeError("llm_config must be an LLMConfig instance, a string, or a mapping")


class AsyncLLM:
    def __init__(self, config, system_msg: Optional[str] = None):
        """
        Initialize the AsyncLLM with a configuration

        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        self.config = resolve_llm_config(config)
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = LLMCostTracker()

    async def __call__(self, prompt):
        input_price, output_price = self.config.require_pricing()
        message = []
        if self.sys_msg is not None:
            message.append({"content": self.sys_msg, "role": "system"})

        message.append({"role": "user", "content": prompt})

        response = await self.aclient.chat.completions.create(
            model=self.config.model,
            messages=message,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens,
            input_price=input_price,
            output_price=output_price,
        )

        ret = response.choices[0].message.content

        logger.debug(
            f"Token usage: {input_tokens} input + {output_tokens} output = "
            f"{input_tokens + output_tokens} total"
        )
        logger.debug(
            f"Cost: ${usage_record['total_cost']:.6f} "
            f"(${usage_record['input_cost']:.6f} for input, "
            f"${usage_record['output_cost']:.6f} for output)"
        )

        return ret

    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter

        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response

        Returns:
            The formatted response data

        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)

        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)

        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")

        return parsed_data

    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()


def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration

    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config

    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    return AsyncLLM(llm_config)
