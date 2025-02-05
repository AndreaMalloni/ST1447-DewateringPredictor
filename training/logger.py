import logging
import os
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

class LoggerManager:
    """
    Singleton per la gestione dei logger. Crea e restituisce un logger per ciascun file di log richiesto.
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """Garantisce che LoggerManager sia un singleton."""
        if not hasattr(cls, "_instance"):
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_logger(log_filename: Optional[str], log_dir: str = LOG_DIR, enabled: bool = True) -> logging.Logger:
        """
        Restituisce un oggetto logger configurato o un logger "neutro" se logging è disabilitato.

        :param log_filename: Nome del file di log (es: "training.log"). Se None, restituisce un logger neutro.
        :param log_dir: Directory in cui salvare i file di log (default: "logs").
        :param enabled: Flag per abilitare o disabilitare il logging (default: True).
        :return: Oggetto logging.Logger configurato o neutro.
        """
        if not enabled:
            # Logger neutro che ignora i log
            logger = logging.getLogger("null_logger")
            logger.addHandler(logging.NullHandler())
            return logger

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if log_filename not in LoggerManager._instances:
            logger = logging.getLogger(log_filename)
            if not logger.hasHandlers():  # Evita di aggiungere più handler al logger
                logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)

                logger.addHandler(file_handler)

            LoggerManager._instances[log_filename] = logger

        return LoggerManager._instances[log_filename]
