# Dataset Processing Pipeline

## Prerequisites

### ParlerTTS Installation
⚠️ **IMPORTANT**: You must install ParlerTTS from a specific commit to ensure compatibility. Do not use the current version.

```bash
pip install git+https://github.com/huggingface/parler-tts.git@5d0aca9753ab74ded179732f5bd797f7a8c6f8ee
```


## Processing Pipeline for Interrupted Dialogue Dataset

The pipeline consists of two main steps:

### 1. Generate Interrupted Dialogue Data
Run the following script to generate the interrupted dialogue dataset:

```bash
python interrupted_dialogue_for_TTS.py
```

This script processes the Smoltalk dataset and creates interrupted dialogue dataset.

### 2. Convert Dialogues to Audio
After generating the interrupted dialogues, 


```sh
python dialogue_to_audio.py
```

This script converts the processed dialogue dataset into an audio dialogue dataset using ParlerTTS.

## Output
The pipeline will generate:
- Interrupted dialogue text data
- Corresponding audio files for the dialogues

## Audio Processing Pipeline for LibriLight Dataset

### Prerequisites

1. Download LibriLight Large Dataset (3.05T):
```bash
wget https://dl.fbaipublicfiles.com/librilight/data/large.tar
tar -xf large.tar
```

2. Download SpiritLM Checkpoints:
   - Visit https://ai.meta.com/resources/models-and-libraries/spirit-lm-downloads/
   - Request model artifacts and wait for approval email
   - Download and extract checkpoints to your desired location

3. Install SpiritLM Dependencies:
```bash
pip install torch torchaudio
pip install git+https://github.com/facebookresearch/spiritlm.git
```

### Processing Steps

#### 1. Process Audio Files with HuBERT

First, we need to convert audio files into HuBERT tokens using SpiritLM tokenizer:

```bash
# Set your log directory
LOG_DIR="logs/process_libri_large_hubert"

# Run the processing script
bash run_hubert_process.sh $LOG_DIR
```

Key parameters in `hubert_audio_process.py`:
- `--dataset_path`: Path to the LibriLight dataset directory
- `--checkpoint_path`: Path to the SpiritLM checkpoints directory
- `--processed_files_log`: Path to save the list of processed files

The script will:
- Process all FLAC files in the dataset
- Convert them to HuBERT tokens using SpiritLM
- Save tokens as JSON files alongside the original audio files
- Keep track of processed files for resume capability

#### 2. Generate Memory-Mapped Files

After processing all audio files, convert the JSON token files into memory-mapped format:

```bash
python mmap_generation.py \
    --original_dir /path/to/processed/dataset \
    --output_dir /path/to/output/mmap/files \
    --output_prefix libri_large \
    --bos_token_id 500 \
    --eos_token_id 501 \
    --start_token_id 102400 \
    --chunk_size 50000000000
```

Key parameters:
- `--original_dir`: Directory containing the processed JSON token files
- `--output_dir`: Directory to save the memory-mapped files
- `--output_prefix`: Prefix for output files (e.g., "libri_large_chunk0.mmap")
- `--bos_token_id`: Beginning of sequence token ID (default: 500)
- `--eos_token_id`: End of sequence token ID (default: 501)
- `--start_token_id`: Start token ID for audio tokens (default: 102400)
- `--chunk_size`: Number of tokens per chunk (default: 50B tokens)

The script will:
- Scan all JSON files in the input directory
- Combine tokens into large memory-mapped files
- Split into chunks if exceeding the specified chunk size
- Add special tokens (BOS/EOS) around each sequence
