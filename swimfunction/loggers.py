import logging
from swimfunction import FileLocations

def get_vanilla_logger(name, path_to_logfile, quiet=False) -> logging.Logger:
    file_formatter = logging.Formatter(
        '%(asctime)s~%(levelname)s  ~  %(message)s  ~  module:%(module)s~function:%(module)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    file_handler = logging.FileHandler(path_to_logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    if not quiet:
        logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger

def get_console_logger(name, level=logging.DEBUG) -> logging.Logger:
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger

def get_video_processing_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'video_processing.log')

def get_metric_calculation_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'metric_calculation.log')

def get_plotting_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'plotting.log')

def get_qc_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'qc.log')

def get_dlc_import_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'dlc_import.log')

def get_cruise_embedding_logger(name) -> logging.Logger:
    return get_vanilla_logger(
        name, FileLocations.get_experiment_cache_root() / 'cruise_embedding.log')
