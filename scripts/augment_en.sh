echo generate English samples in normal style
STYLE=normal
MODEL=openai
LANGUAGE=english
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate English samples in emotional style
STYLE=emotional
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate English samples in propaganda style
STYLE=propaganda
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE

echo generate English samples in subjective style
STYLE=subjective
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE