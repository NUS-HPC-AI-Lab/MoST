# MoST: Mixing Speech and Text with Modality-Aware Mixture of Experts

<p align="center">
  <strong>A novel multimodal large language model that seamlessly integrates speech and text processing through Modality-Aware Mixture of Experts (MAMoE) architecture.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>


---

## Overview

MoST (Mixture of Speech and Text) is a unified foundation model that seamlessly processes and generates both speech and text modalities within a single, end-to-end architecture. Unlike existing approaches that process diverse modality representations with identical parameters, MoST introduces specialized routing pathways through our proposed **Modality-Aware Mixture of Experts (MAMoE)** architecture.

### Key Features

- **Modality-Aware Routing**: Tokens are directed to modality-appropriate experts based on input type
- **Specialized Expert Groups**: Dedicated text and audio expert groups capture domain-specific patterns
- **Cross-Modal Shared Experts**: Facilitate information transfer between modalities
- **Continuous Audio Processing**: Directly processes continuous audio waveforms via HuBERT encoder
- **End-to-End Speech-Text Tasks**: Supports ASR, TTS, spoken QA, and audio language modeling
- **Fully Open-Source**: Built exclusively on open-source datasets

## Architecture

### Modality-Aware Mixture of Experts (MAMoE)

MAMoE is an enhanced routing mechanism that directs tokens to modality-specific experts based on the token's modality type. Unlike traditional MoE where all tokens compete for all experts, MAMoE ensures:

- **Text tokens** are routed to text-specific experts
- **Audio tokens** are routed to audio-specific experts  
- **Shared experts** remain accessible to all modalities for cross-modal understanding

#### How It Works

1. **Modality Tagging**: Each token is tagged with a modality indicator
   - `0`: Text tokens
   - `1`: Audio tokens

2. **Modality-Aware Routing**: The router (`MAMoEGate`) applies a modality mask to gating scores, constraining expert selection to the appropriate modality group

3. **Top-K Selection**: Standard top-k expert selection proceeds with modality-constrained candidates

4. **Shared Expert Processing**: All tokens are additionally processed by shared experts for cross-modal interaction

### Model Components

- **Audio Encoder**: Frozen HuBERT encoder for continuous audio representation
- **Transformer Decoder**: Adapted from DeepSeek-V2 Lite with MAMoE layers
- **Audio Decoder**: HifiGAN vocoder for speech synthesis

## Installation

```bash
# Clone the repository
git clone https://github.com/NUS-HPC-AI-Lab/MoST.git
cd MoST

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.33.0
- Additional dependencies listed in `requirements.txt`

## Usage

### Configuration

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Load MoST with MAMoE configuration
config = AutoConfig.from_pretrained("path/to/most/config")

# Key MAMoE parameters
# config.use_modality_aware_routing = True
# config.n_routed_experts = 64
# config.n_shared_experts = 2
# config.text_expert_indices = [0, 1, ..., 31]  # First 32 experts for text
# config.audio_expert_indices = [32, 33, ..., 63]  # Last 32 experts for audio

model = AutoModelForCausalLM.from_pretrained("path/to/most/model", config=config)
```

### Building MoST from Scratch

#### Step 1: Convert HuBERT Weights

```bash
python convert_hubert.py \
    --fairseq_path /path/to/hubert_base.pt \
    --output_path /path/to/converted_hubert.pt
```

#### Step 2: Initialize MoST with Pre-trained Weights

```bash
# Modify paths in load_deepseek_weights.py, then run:
python load_deepseek_weights.py
```

#### Step 3: Train MoST

```bash
accelerate launch train/run_clm_no_trainer.py \
    --model_name_or_path /path/to/initialized/MoST \
    --output_dir /path/to/output \
    --train_asr_dirs /path/to/asr/data \
    --train_tts_dirs /path/to/tts/data \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --bf16
```

## Results

### ASR and TTS Performance

MoST achieves competitive performance on automatic speech recognition (ASR) and state-of-the-art results on text-to-speech (TTS) across multiple benchmarks.

| Model | LS-Clean ASR | LS-Clean TTS | LS-Other ASR | LS-Other TTS | VoxPopuli ASR | VoxPopuli TTS | CommonVoice ASR | CommonVoice TTS |
|-------|-------------|--------------|--------------|--------------|---------------|---------------|-----------------|-----------------|
| SpeechGPT | 11.0 | 14.1 | 16.7 | 15.3 | 18.2 | 21.3 | 19.4 | 23.2 |
| AudioLM | 9.5 | 9.2 | 12.0 | 12.1 | 15.0 | 22.1 | 17.6 | 25.1 |
| SpiritLM | 6.0 | 6.7 | 11.0 | 9.5 | 14.3 | 19.4 | 15.4 | 22.4 |
| Moshi | 5.5 | 7.0 | 12.0 | 7.2 | 8.8 | 10.6 | 9.4 | 14.2 |
| Qwen2-Audio | **1.8** | - | 3.6 | - | 7.1 | - | 8.6 | - |
| Phi-4 Multimodal | 2.1 | 4.8 | **3.5** | 4.1 | 6.3 | 11.5 | 9.2 | 10.8 |
| SeamlessM4T-v2 | 3.3 | 6.3 | 4.2 | **5.3** | 7.5 | 10.3 | 10.3 | 12.1 |
| MinMo | **1.8** | 6.7 | 3.9 | 7.5 | 6.7 | 10.9 | 8.0 | 13.5 |
| LLaMA-Omni2 | 3.5 | 10.1 | 4.0 | 9.2 | 9.5 | 12.4 | 11.3 | 17.2 |
| **MoST (Ours)** | 2.0 | **6.0** | 3.7 | 7.2 | **6.2** | **10.1** | **8.4** | **11.5** |

*WER (%) ↓ for ASR, WER (%) ↓ for TTS. Lower is better. Best results in **bold**.*

### Audio Language Modeling Performance

MoST demonstrates strong audio language understanding capabilities across multiple benchmarks.

| Model | sWUGGY | sBLIMP | sTopic-StoryCloze | sStoryCloze | Average |
|-------|--------|--------|-------------------|-------------|---------|
| AudioLM | 71.50 | **64.70** | - | - | - |
| SpeechGPT | 51.82 | 49.75 | 60.13 | 53.13 | 53.71 |
| SpiritLM | 40.14 | 48.28 | 83.32 | 58.95 | 57.67 |
| Moshi | 51.14 | 53.31 | 46.34 | 45.16 | 48.99 |
| Phi-4 Multimodal | 71.84 | 60.21 | 81.55 | 62.39 | 69.00 |
| MinMo | 68.59 | 55.43 | 75.43 | 61.29 | 65.19 |
| LLaMA-Omni2 | 73.21 | 53.59 | 78.21 | 68.55 | 68.39 |
| **MoST (Ours)** | **75.28** | 63.42 | **83.64** | 65.43 | **71.94** |

*Accuracy (%) ↑. Higher is better. Best results in **bold**.*

### Spoken Question Answering Performance

MoST excels in spoken question answering tasks, supporting both speech-to-text (S→T) and speech-to-speech (S→S) settings.

| Model | Llama Q (S→T) | Llama Q (S→S) | TriviaQA (S→T) | TriviaQA (S→S) | WebQ (S→T) | WebQ (S→S) |
|-------|---------------|---------------|----------------|----------------|------------|------------|
| SpeechGPT | 45.2 | 34.2 | 28.4 | 18.5 | 35.1 | 24.3 |
| AudioLM | 38.5 | 25.8 | 22.1 | 10.2 | 30.2 | 18.7 |
| SpiritLM | 58.3 | 45.1 | 38.2 | 24.6 | 42.5 | 31.2 |
| Moshi | 52.1 | 40.3 | 35.6 | 20.7 | 38.4 | 28.5 |
| MinMo | 68.5 | 55.2 | 40.1 | 28.4 | 52.3 | 39.8 |
| LLaMA-Omni2 | 62.4 | 48.7 | 36.8 | 25.1 | 48.6 | 35.2 |
| **MoST (Ours)** | **74.8** | **62.6** | **43.5** | **32.1** | **58.2** | **44.7** |

*Accuracy (%) ↑. Higher is better. Best results in **bold**.*

## Training Recipe

MoST follows a two-stage training protocol:

### Stage 1: Cross-Modal Post-Training
- Initialize from DeepSeek-V2 Lite
- Train on ASR and TTS datasets (LibriHeavy, Common Voice, VoxPopuli)
- Prime modality-specific experts for speech processing

### Stage 2: Mixed Instruction Fine-Tuning
- Fine-tune with speech-text instruction dataset
- Mix with ASR/TTS data to prevent catastrophic forgetting
- Enable complex instruction-following capabilities

| Stage | Steps | Batch Size | Task Mixing |
|-------|-------|------------|-------------|
| Stage 1 | 500k | 512 | ASR 0.4, TTS 0.4, Text LM 0.2 |
| Stage 2 | 10k | 128 | Speech Inst. 0.4, Text Inst. 0.4, ASR 0.1, TTS 0.1 |

## Generated Audio Examples

| Example | Transcript | Audio |
|---------|------------|-------|
| EXP1 | I love to play Golf. | [Listen](./asset/exp_1.wav) |
| EXP2 | Today, we find ourselves at a critical point in the history of scientific breakthroughs. | [Listen](./asset/exp_2.wav) |
| EXP3 | But the rapid acceleration of AI's development has raised concerns about its other effects on the rest of our society. | [Listen](./asset/exp_3.wav) |
| EXP4 | China is a very large country and is the most populous country in the world. | [Listen](./asset/exp_4.wav) |

## Model & Data Release

🚧 **Coming Soon**: Model checkpoints and training datasets are currently being prepared for release. Stay tuned for updates.

## Citation

Citation information will be added upon paper publication.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the National University of Singapore HPC-AI Lab.
