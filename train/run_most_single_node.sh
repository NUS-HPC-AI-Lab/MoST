#!/bin/bash

# Default values for optional parameters
MODEL_PATH=${MODEL_PATH:-""}
OUTPUT_DIR=${OUTPUT_DIR:-""}
TRAIN_ASR_DIRS=${TRAIN_ASR_DIRS:-""}
TRAIN_TTS_DIRS=${TRAIN_TTS_DIRS:-""}
VAL_ASR_DIRS=${VAL_ASR_DIRS:-""}
VAL_TTS_DIRS=${VAL_TTS_DIRS:-""}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
WARMUP_STEPS=${WARMUP_STEPS:-0}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}
SEED=${SEED:-42}

# Get number of GPUs per node
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "==============================================="
echo "Single node multi-GPU training configuration:"
echo "Number of GPUs: $NUM_GPUS"
echo "==============================================="

echo "==============================================="
echo "Training parameters:"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Train ASR directories: $TRAIN_ASR_DIRS"
echo "Train TTS directories: $TRAIN_TTS_DIRS"
echo "Val ASR directories: $VAL_ASR_DIRS"
echo "Val TTS directories: $VAL_TTS_DIRS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Num train epochs: $NUM_TRAIN_EPOCHS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Max train steps: $MAX_TRAIN_STEPS"
echo "Seed: $SEED"
echo "==============================================="

# Command to run the training using Accelerator on a single node
accelerate launch \
  --multi_gpu \
  --num_processes=$NUM_GPUS \
  --mixed_precision="bf16" \
  run_clm_no_trainer.py \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --train_asr_dirs $TRAIN_ASR_DIRS \
  --train_tts_dirs $TRAIN_TTS_DIRS \
  --val_asr_dirs $VAL_ASR_DIRS \
  --val_tts_dirs $VAL_TTS_DIRS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --num_warmup_steps $WARMUP_STEPS \
  --checkpointing_steps $SAVE_STEPS \
  --evaluate_every $EVAL_STEPS \
  --seed $SEED \
  --bf16 \
  --with_tracking \
  ${MAX_TRAIN_STEPS:+"--max_train_steps"} ${MAX_TRAIN_STEPS} \
  "$@"  # Pass any additional arguments 