# Multilingual News Subjectivity
This repository contains our approach for the shared task: CLEF 2023 Task 2 subjectivity detection. We evaluate our method on the datasets in English and Turkish.

You can find more details about the task and the dataset at the [link](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab).

You can check the sample distibution via `python -m app.analysis.stats`

## Installation
We use docker to build the environment and the libraries.

```console
docker build -t news_subjectivity:latest .
```

Create .env file and add your OPENAI key in .env file:

`OPENAI_API_KEY=HEREISYOURAPIKEY`

## How to run
To augment the dataset, modify the script `scripts/augment.sh` and then execute the following code:

```console
docker run --rm --gpus 0 -v /reco/news_subjectivity:/app --name app news_subjectivity:latest bash scripts/augment.sh
```

## Augmentation Styles

To change the paraphrase style, you need to change --style in `scripts/augment.sh`.

**normal**: it paraphrases the text without any style. It is a baseline augmentation method. When this style is on, it only paraphrases the minority class which is subjective in the datasets.