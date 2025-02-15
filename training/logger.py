from ctypes import ArgumentError
from datetime import datetime
import logging
import os
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

class LoggerManager:
    _instances = {
        "Training": None,
        "Processing": None
    }

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_logger(type: str, log_dir: str = LOG_DIR, enabled: bool = True) -> logging.Logger:
        if not enabled:
            # Logger neutro che ignora i log
            logger = logging.getLogger("null_logger")
            logger.addHandler(logging.NullHandler())
            return logger

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if type not in ("Training", "Processing"):
            raise ArgumentError("Invalid logger type.")

        log_filename = f'[{type}] {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

        if not LoggerManager._instances[type]:
            logger = logging.getLogger(log_filename)
            if not logger.hasHandlers():  # Evita di aggiungere più handler al logger
                logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)

                logger.addHandler(file_handler)

            LoggerManager._instances[type] = logger

        return LoggerManager._instances[type]
