MODEL=baseline
LANGUAGE=german
MODEL_PATH=saved_model/${MODEL}/${LANGUAGE}/no_augment
PRETRAINED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
TRAIN_DATASET_PATH=dataset/clef2023/${LANGUAGE}/train.tsv
DEV_DATASET_PATH=dataset/clef2023/${LANGUAGE}/dev.tsv
AUGMENT=no_augment

echo Experiment
echo $MODEL
echo $LANGUAGE

mkdir -p $MODEL_PATH

RESULT_OUTPUT=results/${MODEL}/${LANGUAGE}/${AUGMENT}

mkdir -p $RESULT_OUTPUT

python -m app.models.main \
--model $MODEL \
--pretrained_model $PRETRAINED_MODEL \
--model_path $MODEL_PATH \
--dev_dataset_path $DEV_DATASET_PATH \
--result_output $RESULT_OUTPUT/dev.tsv \
--test \
--train_dataset_path $TRAIN_DATASET_PATH \
--class_weight \
--train

echo Evaluation
python -m app.models.evaluate \
--gold-file-path $DEV_DATASET_PATH \
--pred-file-path $RESULT_OUTPUT/dev.tsv

for AUGMENT in normale emotionale Propaganda subjektive all
do
  echo Testing ${LANGUAGE} , augmentation: ${AUGMENT}
  MODEL_PATH=saved_model/${MODEL}/${LANGUAGE}/${AUGMENT}
  AUGMENT_DATASET_PATH=dataset/augmented_data/clef2023/${LANGUAGE}/${AUGMENT}.tsv

  mkdir -p $MODEL_PATH

  RESULT_OUTPUT=results/${MODEL}/${LANGUAGE}/${AUGMENT}

  mkdir -p $RESULT_OUTPUT

  python -m app.models.main \
  --model $MODEL \
  --pretrained_model $PRETRAINED_MODEL \
  --model_path $MODEL_PATH \
  --dev_dataset_path $DEV_DATASET_PATH \
  --result_output $RESULT_OUTPUT/dev.tsv \
  --augment_dataset_path $AUGMENT_DATASET_PATH \
  --test \
  --train_dataset_path $TRAIN_DATASET_PATH \
  --train

  echo Evaluation
  python -m app.models.evaluate \
  --gold-file-path $DEV_DATASET_PATH \
  --pred-file-path $RESULT_OUTPUT/dev.tsv

done



