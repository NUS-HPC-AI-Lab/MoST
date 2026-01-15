import os
import json
import torchaudio
from tqdm import tqdm

# Set SpiritLM checkpoint path
spiritlm_checkpoint_path = "path/to/spiritlm/checkpoints"
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = spiritlm_checkpoint_path
from spiritlm.speech_tokenizer import spiritlm_base

def process_librispeech_split(split_name, output_file, spiritlm_tokenizer):
    # Initialize LibriSpeech dataset
    librispeech_dir = "path/to/librispeech_asr"
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=librispeech_dir,
        url=split_name,
        download=False
    )
    
    instruction_data = []
    
    for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        # Get audio sample and transcript
        waveform, sample_rate, text, speaker_id, chapter_id, utterance_id = dataset[idx]
        
        # Resample if necessary
        if sample_rate != spiritlm_tokenizer.expected_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, spiritlm_tokenizer.expected_sample_rate)
        
        # Get SpiritLM tokens
        spiritlm_tokens = spiritlm_tokenizer.encode_string(waveform)
        
        # Convert tokens to string format
        token_str = spiritlm_tokens
        
        # Create sample
        sample = {
            "id": f"{split_name}_{speaker_id}_{chapter_id}_{utterance_id}",
            "instruction": "Transcribe the following speech into text.",
            "input": token_str,
            "output": text,
            "metadata": {
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "split": split_name
            }
        }
        
        instruction_data.append(sample)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(instruction_data, f, indent=2)
    
    return len(instruction_data)

def main():
    # Initialize SpiritLM tokenizer
    print("Initializing SpiritLM tokenizer...")
    spiritlm_tokenizer = spiritlm_base()
    
    # Process each split
    splits = ['train-clean-100', 'train-clean-360', 'train-other-500', 
              'dev-clean', 'dev-other', 'test-clean', 'test-other']
    
    total_samples = 0
    output_dir = "path/to/librispeech_asr_instructions"
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        output_file = os.path.join(output_dir, f"{split}_instructions.json")
        print(f"\nProcessing {split}...")
        num_samples = process_librispeech_split(split, output_file, spiritlm_tokenizer)
        total_samples += num_samples
        print(f"Processed {num_samples} samples from {split}")
    
    print(f"\nTotal processed samples: {total_samples}")

if __name__ == "__main__":
    main() 