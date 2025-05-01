#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:4
#SBATCH -c 8
#SBATCH --output train_baseline.log
#SBATCH --mem 200gb
set -e
#source activate hurtfulwords

BASE_DIR="/c/Users/noahz/Downloads/HurtfulWords"
OUTPUT_DIR="/c/Users/noahz/Downloads/HurtfulWords/data"
SCIBERT_DIR="/c/Users/noahz/Downloads/baseline_clinical_BERT_1_epoch_512"
mkdir -p "$OUTPUT_DIR/models/"

cd "$BASE_DIR/scripts" 

python finetune_on_pregenerated.py \
	--pregenerated_data "$OUTPUT_DIR/pregen_epochs/128/" \
	--output_dir "$OUTPUT_DIR/models/baseline_clinical_BERT_1_epoch_128/" \
	--bert_model "$SCIBERT_DIR" \
	--do_lower_case \
	--epochs 1 \
	--train_batch_size 16\
	--seed 123

python finetune_on_pregenerated.py \
	--pregenerated_data "$OUTPUT_DIR/pregen_epochs/512/" \
	--output_dir "$OUTPUT_DIR/models/baseline_clinical_BERT_1_epoch_512/" \
	--bert_model "$OUTPUT_DIR/models/baseline_clinical_BERT_1_epoch_128/" \
	--do_lower_case \
	--epochs 1 \
	--train_batch_size 16\
	--seed 123

rm -rf "$OUTPUT_DIR/models/baseline_clinical_BERT_1_epoch_128/"
