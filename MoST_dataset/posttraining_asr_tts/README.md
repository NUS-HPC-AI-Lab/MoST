# ASR to TTS Dataset Converter

This tool converts Automatic Speech Recognition (ASR) instruction datasets to Text-to-Speech (TTS) instruction datasets by swapping the input and output fields and changing the instruction.

## Overview

The ASR datasets have the following format:
```json
{
  "id": "dataset_split_id",
  "instruction": "Transcribe the following speech into text.",
  "input": "[Hu7][Hu90][Hu481]...",  // Speech tokens
  "output": "TRANSCRIBED TEXT",
  "metadata": { ... }
}
```

The converted TTS datasets will have the following format:
```json
{
  "id": "tts_dataset_split_id",
  "instruction": "Convert the following text into speech.",
  "input": "TRANSCRIBED TEXT",  // Now the input is text
  "output": "[Hu7][Hu90][Hu481]...",  // Now the output is speech tokens
  "metadata": {
    ...,
    "task": "tts",
    "original_task": "asr",
    "dataset": "dataset_name",
    "conversion_timestamp": "2023-06-15T12:34:56.789"
  }
}
```

## Usage

### Single Dataset Conversion

To convert a single ASR dataset to TTS:

```bash
python asr_to_tts_converter.py --input_dir /path/to/asr/dataset --output_dir /path/to/tts/output --dataset_name dataset_name
```

Optional arguments:
- `--file_pattern`: Pattern to filter JSON files (e.g., 'train*.json')

### Multiple Datasets Conversion

To convert multiple ASR datasets, create a JSON configuration file (see `sample_config.json`) and run:

```bash
python asr_to_tts_converter.py --config /path/to/config.json
```

### Sample Configuration File

```json
[
  {
    "dataset_name": "librispeech",
    "input_dir": "/path/to/librispeech_asr_instructions",
    "output_dir": "/path/to/librispeech_tts_instructions",
    "file_pattern": "*.json"
  },
  {
    "dataset_name": "common_voice",
    "input_dir": "/path/to/common_voice_asr_instructions",
    "output_dir": "/path/to/common_voice_tts_instructions",
    "file_pattern": "*.json"
  }
]
```

## Examples

### Convert LibriSpeech ASR to TTS

```bash
python asr_to_tts_converter.py --input_dir /home/svu/e0572481/scratch/librispeech_asr_instructions --output_dir /home/svu/e0572481/scratch/librispeech_tts_instructions --dataset_name librispeech
```

### Convert Common Voice ASR to TTS

```bash
python asr_to_tts_converter.py --input_dir /home/svu/e0572481/scratch/common_voice_asr_instructions --output_dir /home/svu/e0572481/scratch/common_voice_tts_instructions --dataset_name common_voice
```

### Convert Multiple Datasets Using Config

```bash
python asr_to_tts_converter.py --config sample_config.json
```

## Notes

- The script handles both list and dictionary JSON formats
- Error handling is included to skip problematic files or samples
- Detailed logging is provided to track the conversion process
- The original metadata is preserved, with additional TTS-specific metadata added 