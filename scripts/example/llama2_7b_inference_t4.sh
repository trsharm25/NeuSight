#!/bin/bash
cd "$(dirname "$0")"

python3 ../pred.py \
    --predictor_name neusight \
    --predictor_path ../asplos/data/predictor/MLP_WAVE \
    --device_config_path '../asplos/data/device_configs/Tesla_T4.json' \
    --model_config_path ../asplos/data/DLmodel_configs/llama2_7b.json \
    --sequence_length 2048 \
    --batch_size 2 \
    --execution_type inf \
    --tile_dataset_dir ../asplos/data/dataset/train \
    --result_dir ./out \