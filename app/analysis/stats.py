import pandas as pd
from pathlib import Path
def stats(dataset_path:str, prefix):
    splits = [f'train_{prefix}.tsv', f'dev_{prefix}.tsv']

    for split in splits:
        dataset_path = Path(dataset_path)
        data = pd.read_csv(dataset_path/split, sep='\t')
        print(f'Number of instances {len(data)} in {split}')
        print(data.groupby(['label']).count())
        if 'language' in data.columns:
            print(data.groupby(['label', 'language']).count())

if __name__ == '__main__':
    # print('Stats for the multilingual dataset')
    # stats(dataset_path='clef2023-checkthat-lab/task2/data/subtask-2-multilingual/processed', prefix='ml')

    print('Stats for the turkish dataset')
    stats(dataset_path='clef2023-checkthat-lab/task2/data/subtask-2-turkish', prefix='tr')

    print('Stats for the english dataset')
    stats(dataset_path='clef2023-checkthat-lab/task2/data/subtask-2-english', prefix='en')