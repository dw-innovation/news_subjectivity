import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import sklearn.externals
from pathlib import Path
import joblib
from loguru import logger

logger.add(f"{__name__}.log", rotation="500 MB")


class Baseline:

    def __init__(self, model_name, pretrained_model, class_weight: None):
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.class_weight = class_weight
        self.encoder = SentenceTransformer(self.pretrained_model)

    def train(self, dataset_path, model_path, augment_dataset_path:None, training_args:None):
        train_data = pd.read_csv(dataset_path, sep='\t')

        if augment_dataset_path:
            augmented_data = pd.read_csv(augment_dataset_path, sep='\t')
            augmented_data.rename(columns={"generated_text": "sentence"}, inplace=True)
            augmented_data["label"] = augmented_data["sentence"].apply(lambda x: "SUBJ")
            augmented_data = augmented_data[['sentence','label']]
            train_data = train_data[['sentence', 'label']]
            train_data = pd.concat([train_data, augmented_data])
            logger.info(f'Number of the documents: {len(train_data)}')
            train_data.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
            logger.info(f'Number of the documents: {len(train_data)}')

        model = LogisticRegression(class_weight="balanced") if self.class_weight else LogisticRegression()

        model.fit(X=self.encoder.encode(train_data['sentence'].values), y=train_data['label'].values)
        joblib.dump(model, Path(model_path) / f"{self.model_name}.pkl")

    def test(self, dataset_path, model_path, result_output):
        test_data = pd.read_csv(dataset_path, sep='\t')

        model = joblib.load(Path(model_path) / f"{self.model_name}.pkl")

        predictions = model.predict(X=self.encoder.encode(test_data['sentence'].values)).tolist()
        pred_df = pd.DataFrame()
        pred_df['sentence_id'] = test_data['sentence_id']
        pred_df['label'] = predictions

        pred_df.to_csv(result_output, index=False, sep='\t')

    def inference(self, dataset):
        pass
