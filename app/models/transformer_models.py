import pandas as pd
import torch
import os
import random
import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments, Trainer, TextClassificationPipeline,
                          EvalPrediction)

logger.add(f"{__name__}.log", rotation="500 MB")


class SubjectivityDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length: int = 128):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        input_text = sample["sentence"]
        label = 1 if sample["label"] == "SUBJ" else 0

        source_encodings = self.tokenizer.batch_encode_plus([input_text], max_length=self.source_max_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            padding="max_length", return_tensors='pt',
                                                            return_token_type_ids=False)

        return dict(
            input_ids=source_encodings['input_ids'].squeeze(0),
            attention_mask=source_encodings['attention_mask'].squeeze(0),
            labels=torch.LongTensor([label]),
        )


def compute_accuracy(p: EvalPrediction):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    map_weighted = average_precision_score(
        y_true=labels, y_score=preds, average='weighted'),
    map_macro = average_precision_score(y_true=labels, y_score=preds),
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"map_weighted": map_weighted, "map_macro": map_macro,
            "f1-score": f1}

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

class XLMR:

    def __init__(self, model_name, pretrained_model, class_weight: None):
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.class_weight = class_weight

    def train(self, train_dataset_path, eval_dataset_path, model_path, augment_dataset_path: None, training_args: None, augment_type: None, device:"0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        set_random_seed(0)
        train_data = pd.read_csv(train_dataset_path, sep='\t')

        if augment_dataset_path:

            if augment_type == 'oversampling':
                augmented_data = pd.read_csv(augment_dataset_path, sep='\t')
                augmented_data.rename(columns={"generated_text": "sentence"}, inplace=True)
                augmented_data["label"] = augmented_data["sentence"].apply(lambda x: "SUBJ")
                augmented_data = augmented_data[['sentence', 'label']]
                train_data = train_data[['sentence', 'label']]
                train_data = pd.concat([train_data, augmented_data])
                logger.info(f'Number of the documents: {len(train_data)}')
                train_data.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
                logger.info(f'Number of the documents: {len(train_data)}')
            elif augment_type == 'undersampling':
                augmented_data = pd.read_csv(augment_dataset_path, sep='\t')
                augmented_data.rename(columns={"generated_text": "sentence"}, inplace=True)
                augmented_data["label"] = augmented_data["sentence"].apply(lambda x: "SUBJ")
                

                num_subj_samples = len(train_data[train_data["label"] == 'SUBJ'])
                num_obj_samples = len(train_data[train_data["label"] == 'OBJ'])

                num_augmented_samples = (num_obj_samples - num_subj_samples) // 2

                if "normal" in augment_dataset_path:
                    # drop samples labeled as obj
                    train_data.drop(train_data[train_data["label"] == 'OBJ'].sample(n=num_augmented_samples, random_state=0).index, inplace=True)

                else:
                    random.seed(0)

                    indices = random.sample(range(0, (num_obj_samples - num_subj_samples) - 1), num_augmented_samples)
                    for i in indices:
                        org = augmented_data.loc[i]["original"]
                        train_data.loc[train_data["sentence"] == org, "sentence"] = augmented_data.loc[i]["sentence"]
                        train_data.loc[train_data["sentence"] == org, "label"] = 'SUBJ'

                logger.info(train_data.groupby(["label"])["sentence"])

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        train_dataset = SubjectivityDataset(
            data=train_data,
            tokenizer=tokenizer
        )

        eval_data = pd.read_csv(eval_dataset_path, sep='\t')

        eval_dataset = SubjectivityDataset(
            data=eval_data,
            tokenizer=tokenizer
        )

        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=2,
            label2id={"OBJ": 0, "SUBJ": 1},
            id2label={0: "OBJ", 1: "SUBJ"}
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            config=config,
        )

        training_args = TrainingArguments(
            learning_rate=training_args["learning_rate"],
            num_train_epochs=training_args["num_train_epochs"],
            evaluation_strategy="steps",
            per_device_train_batch_size=training_args["train_batch"],
            per_device_eval_batch_size=training_args["eval_batch"],
            output_dir=model_path,
            overwrite_output_dir=True,
            do_eval=True,
            do_train=True,
            remove_unused_columns=False,
            warmup_steps=500,
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1-score',
            seed=0,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_accuracy
        )

        trainer.train()

        trainer.save_model(model_path)  # Saves the tokenizer too for easy upload

        # remove the model
        del model
        del trainer
        torch.cuda.empty_cache()


    def test(self, dataset_path, model_path, result_output):
        set_random_seed(0)
        test_data = pd.read_csv(dataset_path, sep='\t')

        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=2,
            label2id={"OBJ": 0, "SUBJ": 1},
            id2label={0: "OBJ", 1: "SUBJ"}
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

        predictions = []
        for idx, row in test_data.iterrows():
            score = pipe(row["sentence"])[0][0]["score"]
            pred = "OBJ" if score >= 0.5 else "SUBJ"
            predictions.append(pred)

        pred_df = pd.DataFrame()
        pred_df['sentence_id'] = test_data['sentence_id']
        pred_df['label'] = predictions

        pred_df.to_csv(result_output, index=False, sep='\t')


def inference(self, dataset):
    set_random_seed(0)
    pass
