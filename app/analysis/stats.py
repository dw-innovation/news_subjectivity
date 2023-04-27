import pandas as pd
from pathlib import Path
from loguru import logger

logger.add(f"{__name__}.log", rotation="500 MB")


def stats(dataset_path: str):

    for file_path in dataset_path.rglob('*/*.tsv'):
        logger.info(f'The stats for {file_path}')
        data = pd.read_csv(file_path, sep='\t')
        logger.info(f'Number of samples {len(data)}')
        logger.info(data.groupby(['label'])['label'].count())


if __name__ == '__main__':
    dataset_path = Path('dataset/clef2023')

    stats(dataset_path)
