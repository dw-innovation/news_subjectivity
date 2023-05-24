import csv
import pandas as pd
from pathlib import Path


def stats(dataset_path: str):
    dataset_path = Path(dataset_path)

    langs = {
        'arabic': 'ar',
        'dutch': 'nl',
        'english': 'en',
        'german': 'de',
        'italian': 'it',
        'turkish': 'tr'
    }

    modes = ['train', 'dev']

    for mode in modes:
        data = pd.read_csv(dataset_path / 'subtask-2-multilingual' / f'{mode}_ml.tsv', sep="\t", quoting=csv.QUOTE_NONE)

        for idx, row in data.iterrows():

            sent_id = row["sentence_id"]

            for lang in langs.keys():

                lang_short = langs[lang]

                for ref_mode in modes:

                    ref_data = pd.read_csv(dataset_path / f'subtask-2-{lang}' / f'{ref_mode}_{lang_short}.tsv',
                                           sep="\t")

                    if sent_id in ref_data['sentence_id'].tolist():
                        data.loc[idx, "language"] = lang
                        continue

        data['language'].fillna(value="turkish", inplace=True)
        data.to_csv(dataset_path / 'subtask-2-multilingual' / 'processed' / f'{mode}_ml.tsv', index=False, sep='\t')


if __name__ == '__main__':
    stats(dataset_path='clef2023-checkthat-lab/task2/data/')
