import logging
from pathlib import Path
from typing import Optional

import yaml


def get_configuration() -> dict:
    """
    Loads the config.yaml file as a dictionary, preserving its nested structure.

    :return: dict: The configuration settings from config.yaml.
    """
    home_dir = Path.home()
    config_file = home_dir / ".ticker_api" / "config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


def set_configuration():
    """
    Creates the `config.yaml` file in the user's home directory under `.ticker_api` if it doesn't exist,
    copying default settings from `config.default.yaml`. Also creates other required directories for the app.

    :return: None
    """
    home_dir = Path.home()
    config_dir = home_dir / ".ticker_api"
    config_file = config_dir / "config.yaml"
    default_config_file = Path(__file__).parent / "config.default.yaml"

    # Create the .ticker_api directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if the config file does not exist
    if not config_file.exists():
        # Read the content from the default configuration file
        with open(default_config_file, "r") as src:
            config_data = src.read()

        # Write the content to the new configuration file
        with open(config_file, "w") as dst:
            dst.write(config_data)

    # Create required directories
    logs_dir = home_dir / ".ticker_api" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)


def get_logger() -> logging.Logger:
    """
    Creates and configures a logger for the 'ticker_api' module. If the logger
    does not have any handlers, it sets up a StreamHandler with a specified format.

    :return: logging.Logger: Configured logger instance for 'ticker_api'.
    """
    logger = logging.getLogger("ticker_api")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def configure_logger(
    external_logger: Optional[logging.Logger] = None,
) -> logging.Logger:
    """
    Configures the global logger by either using an external logger if provided,
    or by creating a new logger using the 'get_logger' function.

    :param external_logger: An externally provided logger instance. If None, a new logger will be created.
    :return: logging.Logger: The configured logger instance.
    """
    global logger
    if external_logger:
        logger = external_logger
    else:
        logger = get_logger()
    return logger


logger = get_logger()
