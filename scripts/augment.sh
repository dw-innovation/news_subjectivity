echo generate Turkish samples in normal style
STYLE=normal
MODEL=openai
LANGUAGE=turkish
OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/

mkdir -p ${OUTPUT_DIR}

python -m app.augmentation \
--model "openai" \
--dataset dataset/clef2023/turkish/train.tsv \
--style $STYLE \
--device 0 \
--output_dir $OUTPUT_DIR \
--language $LANGUAGE