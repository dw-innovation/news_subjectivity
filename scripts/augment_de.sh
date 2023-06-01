echo generate German samples in normal style

MODEL=openai
MODEL_NAME="gpt-3.5-turbo"
LANGUAGE=german
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/chat-gpt

mkdir -p ${OUTPUT_DIR}

for AUGMENT in normale emotionale abwertende Propaganda subjektive parteiische übertriebene voreingenommene ; do

STYLE=${AUGMENT}
echo $MODEL_NAME
python -m app.augmentation \
--model $MODEL \
--model_name $MODEL_NAME \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

done