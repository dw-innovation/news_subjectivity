#echo generate Turkish samples in normal style
#STYLE=normal
#MODEL=openai
#LANGUAGE=turkish
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE
#
#echo generate Turkish samples in emotional style
#STYLE=duygusal
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE
#
#echo generate Turkish samples in propaganda style
#STYLE=propaganda
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE
#
#echo generate Turkish samples in subjective style
#STYLE=öznel
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE

#echo generate Turkish samples in partizan style
#LANGUAGE=turkish
#STYLE=partizan
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE

#echo generate Turkish samples in prejudiced style
#LANGUAGE=turkish
#STYLE=önyargılı
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#mkdir -p ${OUTPUT_DIR}
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE

#echo generate Turkish samples in exagerrated style
#LANGUAGE=turkish
#STYLE=abartılı
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE
#
#
#echo generate Turkish samples in exagerrated style
#LANGUAGE=turkish
#STYLE=aşağılayıcı
#OUTPUT_DIR=dataset/augmented_data/clef2023/${LANGUAGE}/
#
#python -m app.augmentation \
#--model "openai" \
#--dataset dataset/clef2023/${LANGUAGE}/train.tsv \
#--style $STYLE \
#--device 0 \
#--output_dir $OUTPUT_DIR \
#--language $LANGUAGE
