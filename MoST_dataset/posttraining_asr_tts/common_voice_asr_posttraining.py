import os
import json
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset

# Set SpiritLM checkpoint path
spiritlm_checkpoint_path = "path/to/spiritlm/checkpoints"
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = spiritlm_checkpoint_path
# HF_token: Set your Hugging Face token here or use HF_TOKEN environment variable
os.environ['HF_HOME'] = "path/to/huggingface"
from spiritlm.speech_tokenizer import spiritlm_base

def process_common_voice_split(split_name, output_file, spiritlm_tokenizer):
    print(f"Loading Common Voice {split_name} split...")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split=split_name, trust_remote_code=True)
    
    def process_sample(sample):
        # Get audio data and ensure it's float32
        audio_array = torch.from_numpy(sample['audio']['array']).unsqueeze(0).to(torch.float32)
        sample_rate = sample['audio']['sampling_rate']
        
        # Resample if necessary (on CPU)
        if sample_rate != spiritlm_tokenizer.expected_sample_rate:
            audio_array = torchaudio.functional.resample(audio_array, sample_rate, spiritlm_tokenizer.expected_sample_rate)
        
        # Get SpiritLM tokens (on CPU)
        spiritlm_tokens = spiritlm_tokenizer.encode_string(audio_array)
        
        return {
            "id": f"common_voice_{split_name}_{sample['client_id']}",
            "instruction": "Transcribe the following speech into text.",
            "input": spiritlm_tokens,
            "output": sample['sentence'],
            "metadata": {
                "client_id": sample['client_id'],
                "up_votes": sample['up_votes'],
                "down_votes": sample['down_votes'],
                "age": sample['age'],
                "gender": sample['gender'],
                "accent": sample['accent'],
                "locale": sample['locale'],
                "split": split_name
            }
        }
    
    # Process the dataset using map
    instruction_data = dataset.map(
        process_sample,
        remove_columns=dataset.column_names,
        desc=f"Processing {split_name}",
        num_proc=1  # Start with 1 process for debugging
    )
    
    # Convert to list before saving to JSON
    instruction_list = instruction_data.to_list()
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(instruction_list, f, indent=2)
    
    return len(instruction_data)

def main():
    # Initialize SpiritLM tokenizer
    print("Initializing SpiritLM tokenizer...")
    spiritlm_tokenizer = spiritlm_base()
    
    # Process each split
    splits = ['train']
    
    total_samples = 0
    output_dir = "path/to/common_voice_asr_instructions"
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        output_file = os.path.join(output_dir, f"{split}_instructions.json")
        print(f"\nProcessing {split}...")
        num_samples = process_common_voice_split(split, output_file, spiritlm_tokenizer)
        total_samples += num_samples
        print(f"Processed {num_samples} samples from {split}")
    
    print(f"\nTotal processed samples: {total_samples}")

if __name__ == "__main__":
    main()