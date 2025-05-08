# htm_py/logging_config.py
import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler("nab_alignment.log", mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("HTMModel")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # prevent unittest from muting
