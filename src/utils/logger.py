import logging


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger padronizado para o projeto."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    return logging.getLogger(name)
