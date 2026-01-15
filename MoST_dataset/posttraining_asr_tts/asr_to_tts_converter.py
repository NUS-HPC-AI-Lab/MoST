import os
import json
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("asr_to_tts_converter")

def convert_asr_to_tts(input_dir, output_dir, dataset_name=None, file_pattern=None):
    """
    Convert ASR instruction dataset to TTS instruction dataset by swapping input and output
    and changing the instruction.
    
    Args:
        input_dir: Directory containing ASR instruction JSON files
        output_dir: Directory to save TTS instruction JSON files
        dataset_name: Optional name of the dataset (e.g., 'librispeech', 'common_voice')
                     If None, will try to infer from the input directory name
        file_pattern: Optional pattern to filter JSON files (e.g., 'train*.json')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If dataset_name is not provided, try to infer from input_dir
    if dataset_name is None:
        input_path = Path(input_dir)
        if "librispeech" in input_path.name.lower():
            dataset_name = "librispeech"
        elif "common_voice" in input_path.name.lower():
            dataset_name = "common_voice"
        else:
            dataset_name = "unknown"
    
    logger.info(f"Converting ASR to TTS for dataset: {dataset_name}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get all JSON files in the input directory
    if file_pattern:
        import glob
        json_files = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, file_pattern))]
    else:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}" + 
                      (f" matching pattern {file_pattern}" if file_pattern else ""))
        return 0
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    total_samples = 0
    processed_files = 0
    skipped_files = 0
    
    for json_file in json_files:
        input_file_path = os.path.join(input_dir, json_file)
        
        # Create output file name by adding '_tts' before the extension
        output_file_name = json_file.replace('.json', '_tts.json')
        output_file_path = os.path.join(output_dir, output_file_name)
        
        logger.info(f"Converting {input_file_path} to {output_file_path}...")
        
        try:
            # Read the ASR instruction data
            with open(input_file_path, 'r') as f:
                asr_data = json.load(f)
            
            # Check if it's a list or a dictionary
            if isinstance(asr_data, dict):
                # If it's a dictionary, convert to list of samples
                if "data" in asr_data:
                    asr_data = asr_data["data"]
                else:
                    logger.warning(f"Unexpected JSON format in {json_file}. Expected list or dict with 'data' key.")
                    skipped_files += 1
                    continue
            
            # Convert to TTS instruction data
            tts_data = []
            
            for sample in tqdm(asr_data, desc=f"Converting {json_file}"):
                try:
                    # Create a new sample with swapped input and output
                    tts_sample = {
                        "id": f"tts_{sample['id']}",
                        "instruction": "Convert the following text into speech.",
                        "input": sample["output"],  # Text becomes input
                        "output": sample["input"],  # Speech tokens become output
                        "metadata": sample["metadata"].copy() if "metadata" in sample else {}  # Copy metadata
                    }
                    
                    # Add TTS marker to metadata
                    tts_sample["metadata"]["task"] = "tts"
                    tts_sample["metadata"]["original_task"] = "asr"
                    tts_sample["metadata"]["dataset"] = dataset_name
                    tts_sample["metadata"]["conversion_timestamp"] = datetime.now().isoformat()
                    
                    tts_data.append(tts_sample)
                except KeyError as e:
                    logger.warning(f"Skipping sample due to missing key: {e}")
                    continue
            
            # Save the TTS instruction data
            with open(output_file_path, 'w') as f:
                json.dump(tts_data, f, indent=2)
            
            num_samples = len(tts_data)
            total_samples += num_samples
            processed_files += 1
            logger.info(f"Converted {num_samples} samples from {json_file}")
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {str(e)}")
            skipped_files += 1
            continue
    
    logger.info(f"\nConversion Summary:")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Files processed: {processed_files}")
    logger.info(f"Files skipped: {skipped_files}")
    logger.info(f"Total converted samples: {total_samples}")
    
    return total_samples

def process_multiple_datasets(datasets_config):
    """
    Process multiple ASR datasets based on a configuration dictionary.
    
    Args:
        datasets_config: List of dictionaries with keys 'input_dir', 'output_dir', 
                        'dataset_name' (optional), and 'file_pattern' (optional)
    """
    total_samples = 0
    
    for config in datasets_config:
        input_dir = config['input_dir']
        output_dir = config['output_dir']
        dataset_name = config.get('dataset_name')
        file_pattern = config.get('file_pattern')
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name or 'Unknown'}")
        logger.info(f"{'='*50}")
        
        samples = convert_asr_to_tts(input_dir, output_dir, dataset_name, file_pattern)
        total_samples += samples
    
    logger.info(f"\nTotal converted samples across all datasets: {total_samples}")
    return total_samples

def main():
    parser = argparse.ArgumentParser(description="Convert ASR instruction datasets to TTS instruction datasets")
    parser.add_argument("--input_dir", type=str, 
                        help="Directory containing ASR instruction JSON files")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save TTS instruction JSON files")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset (e.g., 'librispeech', 'common_voice')")
    parser.add_argument("--file_pattern", type=str, default=None,
                        help="Pattern to filter JSON files (e.g., 'train*.json')")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file for processing multiple datasets")
    
    args = parser.parse_args()
    
    if args.config:
        # Process multiple datasets from config file
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            if not isinstance(config, list):
                config = [config]
                
            process_multiple_datasets(config)
        except Exception as e:
            logger.error(f"Error processing config file: {str(e)}")
            return
    elif args.input_dir and args.output_dir:
        # Process a single dataset
        convert_asr_to_tts(args.input_dir, args.output_dir, args.dataset_name, args.file_pattern)
    else:
        parser.print_help()
        logger.error("Either --config or both --input_dir and --output_dir must be provided")

if __name__ == "__main__":
    main() 