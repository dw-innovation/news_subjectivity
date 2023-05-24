echo generate German samples in normal style
STYLE=normale
MODEL=openai
LANGUAGE=german
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate German samples in emotional style
STYLE=emotionale
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate German samples in propaganda style
STYLE=Propaganda
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate German samples in subjective style
STYLE=subjektive
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE