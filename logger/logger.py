import logging

LOGGER_DISABLED = {
    'main': False,
    'memory': False,
    'tourney': False,
    'mcts': False,
    'model': False
}


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger_mcts = setup_logger('logger_mcts', '../run/logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']
