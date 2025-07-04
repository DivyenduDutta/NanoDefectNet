import logging
import os
from datetime import datetime
from typing import Optional


class LoggerConfig:
    """Singleton class for configuring and managing a logger."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        This method implements the singleton pattern to ensure that
        the logger configuration is consistent throughout the application.

        Args:
            cls: The class itself.
        """
        if not cls._instance:
            cls._instance = super(LoggerConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: Optional[str] = "nanodefectnet"):
        """
        Initializes the logger configuration.

        Args:
            name (str): The name of the logger. Default is "nanodefectnet".
        """
        if hasattr(self, "_initialized") and self._initialized:  # type: ignore[has-type]
            return  # Prevent re-init

        self.logger_name = name
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = os.path.join(self.log_dir, f"log_{self.timestamp}.log")

        self._logger = logging.getLogger(self.logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._configure_handlers()

        self._initialized = True

    def _configure_handlers(self) -> None:
        """
        Configures the logger handlers.

        This method sets up file and console handlers for the logger,
        ensuring that all log messages are captured and formatted correctly.

        The log messages will include the level, timestamp, filename, and message.
        The log file will be stored in the 'logs' directory with a timestamped filename.

        The console output will also display the same format.
        The log level for both handlers is set to DEBUG.
        """

        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        formatter = logging.Formatter(
            "%(levelname)s - %(asctime)s - %(filename)s - %(message)s"
        )

        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    @property
    def logger(self):
        """
        Returns the configured logger instance.
        """
        return self._logger
