"""Unit tests for configuration module."""
import os
import pytest
from pathlib import Path
from src.configuration import resolve_config_path


def test_resolve_config_path_exists():
    """Test that resolve_config_path returns a valid path."""
    config_path = resolve_config_path()
    assert config_path is not None
    assert isinstance(config_path, str)
    assert len(config_path) > 0


def test_resolve_config_path_file_exists():
    """Test that the resolved config file exists."""
    config_path = resolve_config_path()
    assert os.path.exists(config_path), \
        f"Config file not found at {config_path}"


def test_resolve_config_path_is_file():
    """Test that the resolved path points to a file."""
    config_path = resolve_config_path()
    assert os.path.isfile(config_path), f"{config_path} is not a file"


def test_config_file_is_yaml():
    """Test that config file has .yaml extension."""
    config_path = resolve_config_path()
    assert config_path.endswith('.yaml') or config_path.endswith('.yml'), \
        f"Config file should be YAML, got {config_path}"
