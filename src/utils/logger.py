import logging
import sys
import os

def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger configuré avec affichage console et sauvegarde fichier.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Fichier
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
