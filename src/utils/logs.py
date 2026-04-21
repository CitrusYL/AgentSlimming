import logging
import os
import sys


LOG_LEVEL_ENV = "AGENT_SLIMMING_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"


def _create_logger(name: str = "AgentSlimming") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(_log_level())
    logger.propagate = False
    return logger


def _log_level() -> int:
    level_name = os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


logger = _create_logger()
