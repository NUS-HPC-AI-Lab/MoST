import os
import json
import torch
import torchaudio
from tqdm import tqdm
import random
from pathlib import Path
import jsonlines
import gzip
from datasets import load_dataset

# Set SpiritLM checkpoint path
spiritlm_checkpoint_path = "path/to/spiritlm/checkpoints"
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = spiritlm_checkpoint_path
os.environ['HF_HOME'] = "path/to/huggingface"
from spiritlm.speech_tokenizer import spiritlm_base

def load_libriheavy_manifests(manifest_dir):
    """Load Libriheavy large subset manifest which contains all samples."""
    manifest_file = "libriheavy_cuts_large.jsonl.gz"
    manifest_path = os.path.join(manifest_dir, manifest_file)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    print(f"Loading manifest file: {manifest_path}")
    all_cuts = []
    
    # Open gzipped file and read JSONL
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading manifest"):
            try:
                cut = json.loads(line.strip())
                all_cuts.append(cut)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {str(e)}")
                continue
    
    print(f"Found {len(all_cuts):,} samples in the large subset manifest")
    return all_cuts

def process_audio_file(audio_path, spiritlm_tokenizer):
    """Process a single audio file and return SpiritLM tokens."""
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != spiritlm_tokenizer.expected_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, spiritlm_tokenizer.expected_sample_rate)
    
    # Get SpiritLM tokens
    spiritlm_tokens = spiritlm_tokenizer.encode_string(waveform)
    
    return spiritlm_tokens

def create_asr_dataset(manifest_dir, output_dir):
    """Create ASR dataset from Libriheavy manifests."""
    print("\n=== Starting Libriheavy ASR Dataset Creation ===")
    
    print("\nInitializing SpiritLM tokenizer...")
    spiritlm_tokenizer = spiritlm_base()
    
    print("\nLoading Libriheavy manifests...")
    all_cuts = load_libriheavy_manifests(manifest_dir)
    
    print(f"\nProcessing all {len(all_cuts):,} samples...")
    instruction_data = []
    error_count = 0
    
    for cut in tqdm(all_cuts, desc="Processing audio files"):
        try:
            # Get audio file path
            audio_path = cut['recording']['sources'][0]['source']
            path_prefix = "path/to/libriheavy/"
            audio_path = os.path.join(path_prefix, audio_path)
            
            # Process audio to get SpiritLM tokens
            spiritlm_tokens = process_audio_file(audio_path, spiritlm_tokenizer)
            
            # Get text from the first supervision (original text with casing and punctuation)
            text = cut['supervisions'][0]['custom']['texts'][0]
            
            # Create instruction data
            instruction_data.append({
                "id": cut['id'],
                "instruction": "Transcribe the following speech into text.",
                "input": spiritlm_tokens,
                "output": text,
                "metadata": {
                    "speaker": cut['supervisions'][0]['speaker'],
                    "duration": cut['duration'],
                    "start": cut['start'],
                    "channel": cut['channel']
                }
            })
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {cut['id']}: {str(e)}")
            continue
    
    # Save to JSON file
    output_file = os.path.join(output_dir, "libriheavy_asr_instructions.json")
    print(f"\nSaving processed data to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(instruction_data, f, indent=2)
    
    print("\n=== Processing Complete ===")
    print(f"Successfully processed: {len(instruction_data):,} samples")
    print(f"Failed to process: {error_count:,} samples")
    print(f"Output saved to: {output_file}")

def main():
    manifest_dir = "path/to/libriheavy/"
    output_dir = "path/to/libriheavy_asr_instructions"
    os.makedirs(output_dir, exist_ok=True)
    
    create_asr_dataset(manifest_dir, output_dir)

if __name__ == "__main__":
    main() 