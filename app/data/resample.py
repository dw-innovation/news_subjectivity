from pathlib import Path
import pandas as pd

# resample for qualitative analysis

RANDOM_SEED=0
SAMPLE= 10

output_dir_path = 'dataset/qualitative'
input_dir_path = 'dataset/augmented_data/clef2023'
languages = ['english', 'turkish', 'german']

# gpt_3
for language in languages:
    datasets = (Path(input_dir_path)/language).glob('*.tsv')

    for dataset in datasets:
        if 'all.tsv' in dataset.name:
            continue
        else:
            df = pd.read_csv(dataset, sep='\t')
            df = df.sample(n=SAMPLE, random_state=RANDOM_SEED)
            if 'chat-gpt' == dataset.parent.name:
                df.to_csv(f'{output_dir_path}/{language}_chatgpt_{dataset.name}', sep='\t')
            else:
                df.to_csv(f'{output_dir_path}/{language}_{dataset.name}', sep='\t')

# chat-gpt