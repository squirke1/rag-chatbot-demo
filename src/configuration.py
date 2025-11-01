"""
Configuration helpers for selecting environment-specific YAML files.

This centralizes logic for discovering which configuration file to use based
on explicit paths, environment variables, or sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_ENV = "dev"
ENV_VAR = "RAG_ENV"
PATH_VAR = "RAG_CONFIG_PATH"
CONFIG_DIR_NAME = "configs"
CONFIG_TEMPLATE = "rag.{env}.yaml"
LEGACY_FILENAME = "rag.yaml"


def _project_root() -> Path:
    """Return the repository root (one level above src/)."""
    return Path(__file__).resolve().parent.parent


def _config_dir() -> Path:
    """Return the directory containing configuration files."""
    return _project_root() / CONFIG_DIR_NAME


def _resolve_explicit_path(path_str: str) -> Path:
    """Resolve an explicitly provided config path and validate it exists."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Configuration file not found at {candidate}")
        return candidate

    # First try relative to project root
    root_candidate = _project_root() / candidate
    if root_candidate.exists():
        return root_candidate

    # Fall back to resolving relative to current working directory
    resolved = candidate.resolve()
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Configuration file not found at '{path_str}'. "
        f"Checked {_project_root() / candidate} and {resolved}."
    )


def resolve_config_path(config_path: Optional[str] = None) -> str:
    """
    Determine which configuration file should be used.

    Resolution order:
    1. Explicit `config_path` argument.
    2. `RAG_CONFIG_PATH` environment variable.
    3. `{CONFIG_DIR}/rag.<env>.yaml` where env comes from `RAG_ENV` (defaults to dev).
    4. Legacy `{CONFIG_DIR}/rag.yaml`.

    Returns:
        String path to an existing configuration file.

    Raises:
        FileNotFoundError: If no suitable configuration file can be located.
    """
    # 1. Explicit argument
    if config_path:
        return str(_resolve_explicit_path(config_path))

    # 2. Environment variable override
    env_override = os.getenv(PATH_VAR)
    if env_override:
        return str(_resolve_explicit_path(env_override))

    # 3. Environment-specific configuration
    environment = os.getenv(ENV_VAR, DEFAULT_ENV).strip().lower()
    env_candidate = _config_dir() / CONFIG_TEMPLATE.format(env=environment)
    if env_candidate.exists():
        return str(env_candidate)

    if environment != DEFAULT_ENV:
        raise FileNotFoundError(
            f"Configuration for environment '{environment}' was not found at "
            f"{env_candidate}. Set {PATH_VAR} to point to a valid file or "
            f"create {env_candidate.name}."
        )

    # 4. Legacy fallback (rag.yaml)
    legacy_candidate = _config_dir() / LEGACY_FILENAME
    if legacy_candidate.exists():
        return str(legacy_candidate)

    raise FileNotFoundError(
        "No configuration file found. Expected one of the following:\n"
        f"- Explicit path via argument or {PATH_VAR}\n"
        f"- Environment-based file at {env_candidate}\n"
        f"- Legacy file at {legacy_candidate}"
    )

