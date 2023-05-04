MODEL=xlm-roberta
BATCH_SIZE=16
NUM_TRAIN_EPOCHS=3
LANGUAGE=english
TRAIN_DATASET_PATH=dataset/clef2023/${LANGUAGE}/train.tsv
DEV_DATASET_PATH=dataset/clef2023/${LANGUAGE}/dev.tsv

for PRETRAINED_MODEL in "xlm-roberta-base" "roberta-base"; do
    echo Experiment
    echo $MODEL
    echo $PRETRAINED_MODEL
    echo $LANGUAGE
    AUGMENT=no_augment
    MODEL_PATH=saved_model/${PRETRAINED_MODEL}/${LANGUAGE}/${AUGMENT}

    mkdir -p $MODEL_PATH

    RESULT_OUTPUT=results/${MODEL}/${LANGUAGE}/${AUGMENT}

    mkdir -p $RESULT_OUTPUT

    echo No augment

    python -m app.models.main \
    --model $MODEL \
    --pretrained_model $PRETRAINED_MODEL \
    --model_path $MODEL_PATH \
    --dev_dataset_path $DEV_DATASET_PATH \
    --result_output $RESULT_OUTPUT/dev.tsv \
    --test \
    --train_dataset_path $TRAIN_DATASET_PATH \
    --learning_rate 2e-5 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --train_batch $BATCH_SIZE \
    --eval_batch $BATCH_SIZE \
    --train \
    --cuda_device 1

    echo Evaluation
    python -m app.models.evaluate \
    --gold-file-path $DEV_DATASET_PATH \
    --pred-file-path $RESULT_OUTPUT/dev.tsv

    for AUGMENT_TYPE in oversampling undersampling ; do
          for AUGMENT in normal emotional propaganda subjective all ; do
              echo $AUGMENT
              echo $AUGMENT_TYPE
              echo Testing ${LANGUAGE} , augmentation: ${AUGMENT}
              MODEL_PATH=saved_model/${PRETRAINED_MODEL}/${LANGUAGE}/${AUGMENT}
              AUGMENT_DATASET_PATH=dataset/augmented_data/clef2023/${LANGUAGE}/${AUGMENT}.tsv
  
              mkdir -p $MODEL_PATH
  
              RESULT_OUTPUT=results/${PRETRAINED_MODEL}/${LANGUAGE}/${AUGMENT}
  
              mkdir -p $RESULT_OUTPUT
  
              python -m app.models.main  \
              --model $MODEL \
              --pretrained_model $PRETRAINED_MODEL \
              --model_path $MODEL_PATH \
              --dev_dataset_path $DEV_DATASET_PATH \
              --result_output $RESULT_OUTPUT/dev.tsv \
              --test \
              --train_dataset_path $TRAIN_DATASET_PATH \
              --augment_dataset_path $AUGMENT_DATASET_PATH \
              --class_weight \
              --learning_rate 2e-5 \
              --num_train_epochs $NUM_TRAIN_EPOCHS \
              --train_batch $BATCH_SIZE \
              --eval_batch $BATCH_SIZE \
              --train \
              --cuda_device 1 \
              --augment_type $AUGMENT_TYPE
  
              echo Evaluation
              python -m app.models.evaluate \
              --gold-file-path $DEV_DATASET_PATH \
              --pred-file-path $RESULT_OUTPUT/dev.tsv

        done

    done

done