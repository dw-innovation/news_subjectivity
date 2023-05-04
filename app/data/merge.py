import random
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    dataset_folder = Path('dataset/augmented_data/clef2023')
    # Merge all the datasets
    languages = ['english', 'turkish', 'german']


    augmented_files = {
        'turkish': ['duygusal.tsv', 'normal.tsv', 'propaganda.tsv', 'öznel.tsv'],
        'english': ['emotional.tsv', 'normal.tsv', 'propaganda.tsv', 'subjective.tsv'],
        'german': ['emotionale.tsv', 'normale.tsv', 'Propaganda.tsv', 'subjektive.tsv']
    }

    for language in languages:
        dataset = []
        main_dataset = pd.read_csv(Path('dataset/clef2023') / language / 'train.tsv', sep='\t')

        num_subj_samples = len(main_dataset[main_dataset["label"] == 'SUBJ'])
        num_obj_samples = len(main_dataset[main_dataset["label"] == 'OBJ'])

        num_augmented_samples = num_obj_samples - num_subj_samples

        random.seed(0)

        indices = [*range(num_augmented_samples)]
        random.shuffle(indices)
        idy = 0
        for idx in indices:
            data = pd.read_csv(dataset_folder / language / augmented_files[language][idy], sep='\t')

            dataset.append(data.iloc[[idx]])

            idy += 1

            if idy == len(augmented_files):
                idy = 0

        merged_data = pd.concat(dataset)
        merged_data.drop_duplicates(subset=['generated_text'], keep='first', inplace=True)
        merged_data.to_csv(dataset_folder / language / 'all.tsv', sep='\t', index=False)
