# MoST: Post-Training for ASR and TTS

This directory contains the training infrastructure for post-training the MoST (Mixture of Speech and Text) model on ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) instruction-following datasets.

## Overview

The training infrastructure enables fine-tuning the MoST model on mixed ASR and TTS datasets. Key features include:

1. **Mixed-modality training**: Simultaneously trains on both ASR and TTS tasks in the same batch
2. **Parallelism support**: Supports both data parallelism and expert parallelism for distributed training
3. **Modality-aware routing**: Uses modality-aware MoE routing to direct tokens to modality-specific experts
4. **Multiple data sources**: Can combine data from multiple ASR and TTS sources
5. **WandB integration**: Comprehensive logging and monitoring with Weights & Biases
6. **HuggingFace compatibility**: Works with HuggingFace-hosted models
7. **Checkpoint resume capability**: Resume training from a checkpoint, including optimizer and scheduler states
8. **ASR evaluation metrics**: Includes Word Error Rate (WER) calculation for ASR tasks

## Directory Structure

```
Multimodal-Foundation-Model-with-MoE/
├── datasets/
│   ├── multimodal_dataset.py        # Dataset loader for mixed ASR/TTS data
│   └── posttraining_asr_tts/        # ASR/TTS dataset scripts
│       ├── asr_to_tts_converter.py  # Script to convert ASR data to TTS format
│       ├── common_voice_*.py        # Common Voice dataset processors
│       ├── librispeech_*.py         # LibriSpeech dataset processors
│       └── sample_config.json       # Sample configuration
├── train_most.py                    # Main training script
├── run_training.sh                  # Example training configurations
└── configuration_MoST.py            # MoST model configuration
```

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
wandb
tqdm
numpy
jiwer>=3.0.0  # For WER calculation
```

## Data Format

The training infrastructure expects data in the following JSON format:

### ASR Format
```json
{
  "id": "dataset_split_id",
  "instruction": "Transcribe the following speech into text.",
  "input": "[Hu7][Hu90][Hu481]...",  // Speech tokens
  "output": "TRANSCRIBED TEXT",
  "metadata": { ... }
}
```

### TTS Format
```json
{
  "id": "tts_dataset_split_id",
  "instruction": "Convert the following text into speech.",
  "input": "TEXT TO CONVERT",  
  "output": "[Hu7][Hu90][Hu481]...",  // Speech tokens
  "metadata": { ... }
}
```

## Converting ASR Data to TTS

You can convert ASR datasets to TTS format using the `asr_to_tts_converter.py` script:

```bash
python asr_to_tts_converter.py --input_dir /path/to/asr/dataset \
                             --output_dir /path/to/tts/output \
                             --dataset_name dataset_name
```

See the [ASR to TTS Converter README](./README.md) for more details.

## Training

### Basic Training

```bash
python train_most.py \
    --model_name_or_path username/MoST-deepseek-v2 \
    --train_asr_dirs /path/to/asr/data \
    --train_tts_dirs /path/to/tts/data \
    --val_asr_dirs /path/to/asr/val \
    --val_tts_dirs /path/to/tts/val \
    --output_dir ./outputs \
    --wandb_project most-training \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --fp16 \
    --use_modality_aware_routing \
    --text_expert_indices 0,1,2,3 \
    --audio_expert_indices 4,5,6,7
```

### Resuming Training from a Checkpoint

To resume training from a previously saved checkpoint:

```bash
python train_most.py \
    --model_name_or_path username/MoST-deepseek-v2 \
    --train_asr_dirs /path/to/asr/data \
    --train_tts_dirs /path/to/tts/data \
    --resume_from_checkpoint ./outputs/checkpoint-5000 \
    [other options]
```

If you want to load the model weights but not the optimizer state (useful for fine-tuning):

```bash
python train_most.py \
    --model_name_or_path username/MoST-deepseek-v2 \
    --train_asr_dirs /path/to/asr/data \
    --resume_from_checkpoint ./outputs/checkpoint-5000 \
    --ignore_optimizer_state \
    [other options]
```

### Data Parallel Training

For multi-GPU data parallel training:

```bash
torchrun --nproc_per_node=4 train_most.py \
    --model_name_or_path username/MoST-deepseek-v2 \
    [other options]
```

### Expert Parallel Training

For expert parallel training (spreading experts across GPUs):

```bash
torchrun --nproc_per_node=4 train_most.py \
    --model_name_or_path username/MoST-deepseek-v2 \
    --ep_size 2 \
    [other options]
```

### Using the Run Script

The `run_training.sh` script provides examples for different training configurations:

```bash
./run_training.sh [single|dp|ep|large]
```

- `single`: Run on a single GPU
- `dp`: Run with data parallelism
- `ep`: Run with expert parallelism
- `large`: Run with both data and expert parallelism

## Modality-Aware Routing

The MoST model supports modality-aware routing, which directs tokens to modality-specific experts:

1. Text tokens (token IDs 0-100001) are routed to text-specific experts
2. Audio tokens (token IDs 100002-100503) are routed to audio-specific experts

To enable modality-aware routing, use these command-line arguments:

```bash
--use_modality_aware_routing \
--text_expert_indices 0,1,2,3 \
--audio_expert_indices 4,5,6,7
```

## Monitoring Training

Training progress is logged to Weights & Biases (wandb). Key metrics include:

- Overall training loss
- ASR-specific loss
- TTS-specific loss
- ASR Word Error Rate (WER)
- Learning rate
- Validation perplexity

To access logs, visit your W&B project dashboard at `https://wandb.ai/username/most-training`.

## Implementation Details

### Mixed Batching

The training infrastructure mixes ASR and TTS samples in the same batch based on the `--asr_tts_mix_ratio` parameter (default: 0.5). This helps the model learn both tasks simultaneously.

### Expert Parallelism

Expert parallelism distributes MoE experts across GPUs. In this implementation:

1. World size (total GPUs) is divided into data parallel groups
2. Each data parallel group handles a subset of the batch
3. Within each data parallel group, experts are distributed using expert parallelism
4. Distributed Data Parallel (DDP) communication happens within data parallel groups

### Checkpointing and Resume Capability

The training script automatically saves checkpoints during training:

1. At regular intervals specified by `--save_steps`
2. After validation runs specified by `--eval_steps`
3. At the end of training

Each checkpoint includes:
- Model weights
- Tokenizer configuration
- Optimizer state
- Learning rate scheduler state
- Training arguments and current step

### ASR Evaluation Metrics

For ASR tasks, the evaluation includes:
- Loss and perplexity metrics
- Word Error Rate (WER) using the jiwer library
- Task-specific performance tracking

## Customization

### Adding New Datasets

To add a new dataset source:

1. Process your data into the expected JSON format
2. Add the directory to the `--train_asr_dirs` or `--train_tts_dirs` parameters

### Training Configuration

The training script supports many configuration options:

- Batch size, gradient accumulation
- Learning rate, warmup, scheduler
- Mixed precision (FP16)
- Gradient checkpointing
- Modality-aware routing settings
- Expert and data parallelism options
- Checkpoint and resume options

See `python train_most.py --help` for all available options.

## References

- DeepSeek-V2: [DeepSeek-AI/DeepSeek-V2-Lite](https://huggingface.co/DeepSeek-AI/DeepSeek-V2-Lite)
- Mixture of Experts (MoE): [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Modality-aware routing: [MoST: Mixture of Speech and Text model repository](https://github.com/yourusername/Multimodal-Foundation-Model-with-MoE) 