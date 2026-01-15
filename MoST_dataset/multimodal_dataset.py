import os
import json
import random
import logging
import torch
from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer, AutoTokenizer
import torchaudio
import gzip

# Import the custom tokenizer class
import sys
sys.path.append("path/to/model/MoST-initialized")
from tokenization_most_fast import MoSTTokenizerFast

# Set up environment variables for SpiritLM
spiritlm_checkpoint_path = "path/to/model/spiritlm/checkpoints"
os.environ["SPIRITLM_CHECKPOINTS_DIR"] = spiritlm_checkpoint_path
os.environ['HF_HOME'] = "path/to/model/huggingface"
from spiritlm.speech_tokenizer import spiritlm_base

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("multimodal_dataset")

class MultimodalInstructionDataset(Dataset):
    """
    A dataset class for loading both ASR and TTS instruction datasets.
    
    This class handles mixed batches of ASR and TTS samples, supporting multiple data sources
    for each modality. It loads data from JSON files in the specified directories, with each
    file containing samples in the format:
    
    {
        "id": "dataset_split_id",
        "instruction": "Transcribe the following speech into text.", (for ASR)
                      or "Convert the following text into speech.", (for TTS)
        "input": "[Hu7][Hu90][Hu481]...", (for ASR) or "TEXT" (for TTS),
        "output": "TRANSCRIBED TEXT", (for ASR) or "[Hu7][Hu90][Hu481]..." (for TTS),
        "metadata": { ... }
    }
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        asr_data_dirs: List[str] = None,
        tts_data_dirs: List[str] = None,
        asr_file_patterns: List[str] = None,
        tts_file_patterns: List[str] = None,
        max_seq_length: int = 2048,
        asr_tts_mix_ratio: float = 0.5  # 0.5 means equal number of ASR and TTS samples
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for encoding inputs and outputs
            asr_data_dirs: List of directories containing ASR instruction datasets
            tts_data_dirs: List of directories containing TTS instruction datasets
            asr_file_patterns: List of patterns to filter ASR JSON files (e.g., ["*train*.json", "*dev*.json"])
            tts_file_patterns: List of patterns to filter TTS JSON files (e.g., ["*train*.json", "*dev*.json"])
            max_seq_length: Maximum sequence length for inputs
            asr_tts_mix_ratio: Ratio of ASR to TTS samples (0.5 = equal mix)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.asr_tts_mix_ratio = asr_tts_mix_ratio
        
        # Default patterns if none provided
        if asr_file_patterns is None:
            asr_file_patterns = ["*.json"]
        if tts_file_patterns is None:
            tts_file_patterns = ["*.json"]
            
        # Convert single pattern to list if needed
        if isinstance(asr_file_patterns, str):
            asr_file_patterns = [asr_file_patterns]
        if isinstance(tts_file_patterns, str):
            tts_file_patterns = [tts_file_patterns]
        
        self.asr_samples = []
        self.tts_samples = []
        
        # Load ASR data
        if asr_data_dirs:
            for asr_dir in asr_data_dirs:
                for pattern in asr_file_patterns:
                    self._load_data(asr_dir, "asr", pattern)
                
        # Load TTS data
        if tts_data_dirs:
            for tts_dir in tts_data_dirs:
                for pattern in tts_file_patterns:
                    self._load_data(tts_dir, "tts", pattern)
        
        # Check if we have data
        if not self.asr_samples and not self.tts_samples:
            raise ValueError("No data loaded. Check your data directories and file patterns.")
        
        logger.info(f"Loaded {len(self.asr_samples)} ASR samples and {len(self.tts_samples)} TTS samples.")
        
        # Create combined index mapping for __getitem__
        self._create_index_mapping()
    
    def _load_data(self, data_dir: str, modality: str, file_pattern: str):
        """Load data from directory matching the file pattern."""
        import glob
        
        # if not os.path.exists(data_dir):
        #     logger.warning(f"{modality.upper()} data directory {data_dir} does not exist. Skipping.")
        #     return
        
        # Get all matching files
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        # if not file_paths:
        #     logger.warning(f"No {modality.upper()} files found in {data_dir} with pattern {file_pattern}")
        #     return
        
        # logger.info(f"Loading {modality.upper()} data from {len(file_paths)} files in {data_dir}")
        
        sample_list = self.asr_samples if modality == "asr" else self.tts_samples
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle both list and dict formats
                if isinstance(data, dict):
                    if "data" in data:
                        samples = data["data"]
                    else:
                        samples = [data]
                elif isinstance(data, list):
                    samples = data
                else:
                    logger.warning(f"Unexpected data format in {file_path}. Skipping.")
                    continue
                
                # Add the task type if not present in metadata
                for sample in samples:
                    if "metadata" not in sample:
                        sample["metadata"] = {}
                    if "task" not in sample["metadata"]:
                        sample["metadata"]["task"] = modality
                    
                    # Store dataset source information
                    if "dataset" not in sample["metadata"]:
                        # Try to extract dataset name from file path
                        dataset_name = os.path.basename(data_dir)
                        sample["metadata"]["dataset"] = dataset_name
                
                sample_list.extend(samples)
                # logger.info(f"Loaded {len(samples)} samples from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
    
    def _create_index_mapping(self):
        """Create a mapping from dataset index to (modality, sample_index)."""
        self.index_mapping = []
        
        # Determine how many samples of each type to include
        total_samples = len(self.asr_samples) + len(self.tts_samples)
        
        if total_samples == 0:
            return
        
        if not self.asr_samples:
            # Only TTS samples available
            self.index_mapping = [("tts", i) for i in range(len(self.tts_samples))]
        elif not self.tts_samples:
            # Only ASR samples available
            self.index_mapping = [("asr", i) for i in range(len(self.asr_samples))]
        else:
            # Both ASR and TTS samples available
            # Calculate how many ASR samples to include
            asr_count = int(total_samples * self.asr_tts_mix_ratio)
            asr_count = min(asr_count, len(self.asr_samples))
            
            # Calculate how many TTS samples to include
            tts_count = total_samples - asr_count
            tts_count = min(tts_count, len(self.tts_samples))
            
            # Adjust asr_count if tts_count had to be reduced
            if tts_count < total_samples - asr_count:
                asr_count = min(len(self.asr_samples), total_samples - tts_count)
            
            # Create the index mapping
            self.index_mapping = [("asr", i % len(self.asr_samples)) for i in range(asr_count)] + \
                                [("tts", i % len(self.tts_samples)) for i in range(tts_count)]
            
            # Shuffle the mapping
            random.shuffle(self.index_mapping)
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        modality, sample_idx = self.index_mapping[idx]
        
        if modality == "asr":
            sample = self.asr_samples[sample_idx]
            input_has_audio = True
            output_has_audio = False
        else:  # tts
            sample = self.tts_samples[sample_idx]
            input_has_audio = False
            output_has_audio = True
        
        # Format the sample for model input
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        
        # Check if input contains audio tokens (for TTS input or ASR output)
        input_has_audio = "[Hu" in input_text
        output_has_audio = "[Hu" in output_text
        
        # Wrap audio tokens with begin/end audio tokens if needed
        if input_has_audio and hasattr(self.tokenizer, 'begin_audio_token_id'):
            input_text = f"<|begin_of_audio|>{input_text}<|end_of_audio|>"
        
        if output_has_audio and hasattr(self.tokenizer, 'begin_audio_token_id'):
            output_text = f"<|begin_of_audio|>{output_text}<|end_of_audio|>"
        
        # Format as a chat prompt with BOS and EOS
        prompt = f"<|begin_of_sentence|>{instruction}\n\n{input_text}<|end_of_sentence|>\n\nOutput: "
        
        # Wrap output with BOS and EOS
        response = f"<|begin_of_sentence|>{output_text}<|end_of_sentence|>"
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Tokenize the response
        response_tokens = self.tokenizer.encode(
            response,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Combine prompt and response tokens
        combined_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
        
        # Create attention mask (1 for real tokens)
        attention_mask = torch.ones_like(combined_tokens)
        
        # Truncate if needed (from the beginning if too long)
        start_idx = 0
        if combined_tokens.shape[1] > self.max_seq_length:
            # Keep the most recent tokens (truncate from beginning)
            start_idx = combined_tokens.shape[1] - self.max_seq_length
            combined_tokens = combined_tokens[:, start_idx:]
            attention_mask = attention_mask[:, start_idx:]
        
        # Pad if needed (at the beginning)
        padding_length = 0
        if combined_tokens.shape[1] < self.max_seq_length:
            padding_length = self.max_seq_length - combined_tokens.shape[1]
            # Pad at the beginning
            combined_tokens = torch.nn.functional.pad(
                combined_tokens, (padding_length, 0), value=self.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (padding_length, 0), value=0
            )
        
        # Create labels: -100 for prompt tokens and padding, actual token ids for response
        labels = torch.full_like(combined_tokens, -100)  # Start with all -100
        
        # Calculate where the response starts in the padded sequence
        if padding_length > 0:
            response_start_idx = padding_length + prompt_tokens.shape[1]
        else:
            # If we truncated, we need to check if any prompt tokens remain
            if start_idx >= prompt_tokens.shape[1]:
                # All prompt tokens were truncated
                response_start_idx = 0
            else:
                # Some prompt tokens remain
                response_start_idx = prompt_tokens.shape[1] - start_idx
        
        # Set labels for response tokens
        response_length = min(response_tokens.shape[1], self.max_seq_length - response_start_idx)
        if response_length > 0:
            labels[0, response_start_idx:response_start_idx + response_length] = combined_tokens[0, response_start_idx:response_start_idx + response_length]
        
        return {
            "input_ids": combined_tokens.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "task": modality,
            "sample_id": sample["id"] if "id" in sample else f"{modality}_{sample_idx}"
        }


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    asr_dirs: List[str] = None,
    tts_dirs: List[str] = None,
    train_asr_file_patterns: Union[str, List[str]] = "*train*.json",
    train_tts_file_patterns: Union[str, List[str]] = "*train*.json",
    val_asr_file_patterns: Union[str, List[str]] = ["*dev*.json", "*valid*.json"],
    val_tts_file_patterns: Union[str, List[str]] = ["*dev*.json", "*valid*.json"],
    max_seq_length: int = 2048,
    asr_tts_mix_ratio: float = 0.5,
    train_batch_size: int = 8,
    eval_batch_size: int = 4,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for training and validation.
    
    Args:
        tokenizer: Tokenizer for encoding inputs and outputs
        asr_dirs: List of directories containing ASR instruction datasets
        tts_dirs: List of directories containing TTS instruction datasets
        train_asr_file_patterns: Pattern(s) to filter ASR JSON files for training (e.g., "*train*.json")
        train_tts_file_patterns: Pattern(s) to filter TTS JSON files for training (e.g., "*train*.json")
        val_asr_file_patterns: Pattern(s) to filter ASR JSON files for validation (e.g., ["*dev*.json", "*valid*.json"])
        val_tts_file_patterns: Pattern(s) to filter TTS JSON files for validation (e.g., ["*dev*.json", "*valid*.json"])
        max_seq_length: Maximum sequence length for inputs
        asr_tts_mix_ratio: Ratio of ASR to TTS samples (0.5 = equal mix)
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for validation
        num_workers: Number of workers for dataloaders
        world_size: Number of processes in distributed training
        rank: Rank of the current process
        seed: Random seed
    
    Returns:
        Tuple of (train_dataloader, eval_asr_dataloader, eval_tts_dataloader)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create training dataset
    if asr_dirs or tts_dirs:
        train_dataset = MultimodalInstructionDataset(
            tokenizer=tokenizer,
            asr_data_dirs=asr_dirs,
            tts_data_dirs=tts_dirs,
            asr_file_patterns=train_asr_file_patterns,
            tts_file_patterns=train_tts_file_patterns,
            max_seq_length=max_seq_length,
            asr_tts_mix_ratio=asr_tts_mix_ratio
        )
        
        # Create distributed sampler for training
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed
            )
        else:
            train_sampler = None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_dataloader = None
    
    # Create ASR validation dataset
    eval_asr_dataloader = None
    if asr_dirs:
        val_asr_dataset = MultimodalInstructionDataset(
            tokenizer=tokenizer,
            asr_data_dirs=asr_dirs,
            tts_data_dirs=None, # Only ASR data
            asr_file_patterns=val_asr_file_patterns,
            tts_file_patterns=None,
            max_seq_length=max_seq_length,
            asr_tts_mix_ratio=1.0 # Ensure only ASR is loaded if somehow both dirs were passed
        )

        if len(val_asr_dataset) > 0:
            # Create distributed sampler for ASR validation
            if world_size > 1:
                val_asr_sampler = DistributedSampler(
                    val_asr_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
            else:
                val_asr_sampler = None
            
            eval_asr_dataloader = DataLoader(
                val_asr_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=val_asr_sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            logger.warning("No ASR validation data found or loaded. ASR validation dataloader will be None.")
            eval_asr_dataloader = None

    # Create TTS validation dataset
    eval_tts_dataloader = None
    if tts_dirs:
        val_tts_dataset = MultimodalInstructionDataset(
            tokenizer=tokenizer,
            asr_data_dirs=None, # Only TTS data
            tts_data_dirs=tts_dirs,
            asr_file_patterns=None,
            tts_file_patterns=val_tts_file_patterns,
            max_seq_length=max_seq_length,
            asr_tts_mix_ratio=0.0 # Ensure only TTS is loaded
        )

        if len(val_tts_dataset) > 0:
            # Create distributed sampler for TTS validation
            if world_size > 1:
                val_tts_sampler = DistributedSampler(
                    val_tts_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                )
            else:
                val_tts_sampler = None
            
            eval_tts_dataloader = DataLoader(
                val_tts_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=val_tts_sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            logger.warning("No TTS validation data found or loaded. TTS validation dataloader will be None.")
            eval_tts_dataloader = None

    return train_dataloader, eval_asr_dataloader, eval_tts_dataloader

# New LibriHeavy dataset classes
class LibriHeavyBaseDataset(Dataset):
    """
    Base class for LibriHeavy datasets with lazy loading of audio files.
    
    This handles the common functionality for loading manifest files and 
    processing samples at runtime to avoid loading the entire dataset into memory.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        manifest_dir: str,
        manifest_file: str = "libriheavy_cuts_large.jsonl.gz",
        audio_base_path: str = "path/to/data/libriheavy/",
        max_seq_length: int = 2048,
        max_samples: Optional[int] = None,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42
    ):
        """
        Initialize LibriHeavy dataset.
        
        Args:
            tokenizer: Tokenizer for encoding inputs and outputs
            manifest_dir: Directory containing LibriHeavy manifest files
            manifest_file: Name of the manifest file
            audio_base_path: Base path to prepend to audio file paths
            max_seq_length: Maximum sequence length for inputs
            max_samples: Maximum number of samples to load (None = all)
            split_ratio: Ratio to split between train and validation (0.9 = 90% train)
            is_train: Whether this dataset is for training (True) or validation (False)
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.manifest_dir = manifest_dir
        self.manifest_file = manifest_file
        self.audio_base_path = audio_base_path
        self.max_seq_length = max_seq_length
        self.max_samples = max_samples
        self.is_train = is_train
        self.split_ratio = split_ratio
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize SpiritLM tokenizer
        logger.info("Initializing SpiritLM tokenizer")
        self.spiritlm_tokenizer = spiritlm_base()
        
        # Load manifest entries
        self.samples = self._load_manifest()
        logger.info(f"Loaded {len(self.samples)} samples from manifest")
        
        # Split data if needed
        if split_ratio < 1.0:
            train_size = int(len(self.samples) * self.split_ratio)
            if self.is_train:
                self.samples = self.samples[:train_size]
                logger.info(f"Using {len(self.samples)} samples for training")
            else:
                self.samples = self.samples[train_size:]
                logger.info(f"Using {len(self.samples)} samples for validation")
        
        # Limit samples if max_samples is specified
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            logger.info(f"Limiting to {len(self.samples)} samples")
    
    def _load_manifest(self):
        """Load LibriHeavy manifest file."""
        manifest_path = os.path.join(self.manifest_dir, self.manifest_file)
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        logger.info(f"Loading manifest file: {manifest_path}")
        all_cuts = []
        
        # Open gzipped file and read JSONL
        with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    cut = json.loads(line.strip())
                    all_cuts.append(cut)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON line: {str(e)}")
                    continue
        
        # Shuffle data
        random.shuffle(all_cuts)
        
        return all_cuts
    
    def _get_audio_path(self, cut):
        """Get audio file path from cut."""
        audio_path = cut['recording']['sources'][0]['source']
        return os.path.join(self.audio_base_path, audio_path)
    
    def _get_transcript(self, cut):
        """Get transcript from cut."""
        return cut['supervisions'][0]['custom']['texts'][0]
    
    def _load_audio_waveform(self, audio_path):
        """Load audio waveform from file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.spiritlm_tokenizer.expected_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.spiritlm_tokenizer.expected_sample_rate)
        
        return waveform
    
    def _tokenize_audio(self, waveform):
        """Tokenize audio waveform to SpiritLM tokens."""
        return self.spiritlm_tokenizer.encode_string(waveform)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Should be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement __getitem__")


class LibriHeavyASRWaveDataset(LibriHeavyBaseDataset):
    """
    LibriHeavy ASR dataset with waveform inputs.
    
    This dataset loads audio waveforms and provides them directly as input
    for ASR tasks, with the transcript as output.
    """
    
    def __getitem__(self, idx):
        cut = self.samples[idx]
        
        # Get audio path and transcript
        audio_path = self._get_audio_path(cut)
        transcript = self._get_transcript(cut)
        
        # Load audio waveform
        waveform = self._load_audio_waveform(audio_path)
        
        # Format the sample for model input
        instruction = "Transcribe the following speech into text."
        
        # For waveform input, we'll need to handle differently
        # Here we'd typically have some encoding for the waveform
        # For now, we'll just use it as a placeholder and handle properly in the model
        input_text = f"[AUDIO_WAVEFORM]"  # Placeholder
        output_text = transcript
        
        # Format as a chat prompt with BOS and EOS
        prompt = f"<|begin_of_sentence|>{instruction}\n\n{input_text}<|end_of_sentence|>\n\nOutput: "
        
        # Wrap output with BOS and EOS
        response = f"<|begin_of_sentence|>{output_text}<|end_of_sentence|>"
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Tokenize the response
        response_tokens = self.tokenizer.encode(
            response,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Combine prompt and response tokens
        combined_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
        
        # Create attention mask (1 for real tokens)
        attention_mask = torch.ones_like(combined_tokens)
        
        # Truncate if needed (from the beginning if too long)
        if combined_tokens.shape[1] > self.max_seq_length:
            start_idx = combined_tokens.shape[1] - self.max_seq_length
            combined_tokens = combined_tokens[:, start_idx:]
            attention_mask = attention_mask[:, start_idx:]
        
        # Create labels: -100 for prompt tokens, actual token ids for response
        labels = torch.full_like(combined_tokens, -100)
        labels[0, prompt_tokens.shape[1]:] = combined_tokens[0, prompt_tokens.shape[1]:]
        
        return {
            "input_ids": combined_tokens.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "waveform": waveform.squeeze(0),  # Include the actual waveform
            "task": "asr_wave",
            "sample_id": cut['id']
        }


class LibriHeavyASRTokenDataset(LibriHeavyBaseDataset):
    """
    LibriHeavy ASR dataset with audio token inputs.
    
    This dataset loads audio and converts it to SpiritLM tokens as input
    for ASR tasks, with the transcript as output.
    """
    
    def __getitem__(self, idx):
        cut = self.samples[idx]
        
        # Get audio path and transcript
        audio_path = self._get_audio_path(cut)
        transcript = self._get_transcript(cut)
        
        try:
            # Load audio waveform
            waveform = self._load_audio_waveform(audio_path)
            
            # Tokenize audio to SpiritLM tokens
            spiritlm_tokens = self._tokenize_audio(waveform)
            
            # Format the sample for model input
            instruction = "Transcribe the following speech into text."
            
            # Add begin/end audio tokens around the SpiritLM tokens
            if hasattr(self.tokenizer, 'begin_audio_token_id'):
                input_text = f"<|begin_of_audio|>{spiritlm_tokens}<|end_of_audio|>"
            else:
                input_text = spiritlm_tokens
                
            output_text = transcript
            
            # Format as a chat prompt with BOS and EOS
            prompt = f"<|begin_of_sentence|>{instruction}\n\n{input_text}<|end_of_sentence|>\n\nOutput: "
            
            # Wrap output with BOS and EOS
            response = f"<|begin_of_sentence|>{output_text}<|end_of_sentence|>"
            
            # Tokenize the prompt
            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Tokenize the response
            response_tokens = self.tokenizer.encode(
                response,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Combine prompt and response tokens
            combined_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
            
            # Create attention mask (1 for real tokens)
            attention_mask = torch.ones_like(combined_tokens)
            
            # Truncate if needed (from the beginning if too long)
            start_idx = 0
            if combined_tokens.shape[1] > self.max_seq_length:
                start_idx = combined_tokens.shape[1] - self.max_seq_length
                combined_tokens = combined_tokens[:, start_idx:]
                attention_mask = attention_mask[:, start_idx:]
            
            # Create labels: -100 for prompt tokens, actual token ids for response
            labels = torch.full_like(combined_tokens, -100)
            
            # Calculate where the response starts in the sequence
            if start_idx >= prompt_tokens.shape[1]:
                # All prompt tokens were truncated
                response_start_idx = 0
            else:
                # Some prompt tokens remain
                response_start_idx = prompt_tokens.shape[1] - start_idx
            
            # Set labels for response tokens
            if response_start_idx < combined_tokens.shape[1]:
                labels[0, response_start_idx:] = combined_tokens[0, response_start_idx:]
            
            return {
                "input_ids": combined_tokens.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
                "task": "asr_token",
                "sample_id": cut['id']
            }
        
        except Exception as e:
            logger.warning(f"Error processing sample {cut['id']}: {str(e)}")
            # Return a simple fallback sample in case of errors
            # This allows the dataloader to continue rather than crashing
            dummy_tokens = torch.ones(100, dtype=torch.long) * self.tokenizer.pad_token_id
            return {
                "input_ids": dummy_tokens,
                "attention_mask": torch.zeros_like(dummy_tokens),
                "labels": torch.full_like(dummy_tokens, -100),
                "task": "asr_token",
                "sample_id": cut['id']
            }


class LibriHeavyTTSTokenDataset(LibriHeavyBaseDataset):
    """
    LibriHeavy TTS dataset with transcript inputs and audio token outputs.
    
    This dataset uses transcripts as input and generates SpiritLM tokens
    as output for TTS tasks.
    """
    
    def __getitem__(self, idx):
        cut = self.samples[idx]
        
        # Get audio path and transcript
        audio_path = self._get_audio_path(cut)
        transcript = self._get_transcript(cut)
        
        try:
            # Load audio waveform
            waveform = self._load_audio_waveform(audio_path)
            
            # Tokenize audio to SpiritLM tokens
            spiritlm_tokens = self._tokenize_audio(waveform)
            
            # Format the sample for model input
            instruction = "Convert the following text into speech."
            
            input_text = transcript
            
            # Add begin/end audio tokens around the SpiritLM tokens for output
            if hasattr(self.tokenizer, 'begin_audio_token_id'):
                output_text = f"<|begin_of_audio|>{spiritlm_tokens}<|end_of_audio|>"
            else:
                output_text = spiritlm_tokens
            
            # Format as a chat prompt with BOS and EOS
            prompt = f"<|begin_of_sentence|>{instruction}\n\n{input_text}<|end_of_sentence|>\n\nOutput: "
            
            # Wrap output with BOS and EOS
            response = f"<|begin_of_sentence|>{output_text}<|end_of_sentence|>"
            
            # Tokenize the prompt
            prompt_tokens = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Tokenize the response
            response_tokens = self.tokenizer.encode(
                response,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Combine prompt and response tokens
            combined_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
            
            # Create attention mask (1 for real tokens)
            attention_mask = torch.ones_like(combined_tokens)
            
            # Truncate if needed (from the beginning if too long)
            start_idx = 0
            if combined_tokens.shape[1] > self.max_seq_length:
                start_idx = combined_tokens.shape[1] - self.max_seq_length
                combined_tokens = combined_tokens[:, start_idx:]
                attention_mask = attention_mask[:, start_idx:]
            
            # Create labels: -100 for prompt tokens, actual token ids for response
            labels = torch.full_like(combined_tokens, -100)
            
            # Calculate where the response starts in the sequence
            if start_idx >= prompt_tokens.shape[1]:
                # All prompt tokens were truncated
                response_start_idx = 0
            else:
                # Some prompt tokens remain
                response_start_idx = prompt_tokens.shape[1] - start_idx
            
            # Set labels for response tokens
            if response_start_idx < combined_tokens.shape[1]:
                labels[0, response_start_idx:] = combined_tokens[0, response_start_idx:]
            
            return {
                "input_ids": combined_tokens.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
                "task": "tts_token",
                "sample_id": cut['id']
            }
        
        except Exception as e:
            logger.warning(f"Error processing sample {cut['id']}: {str(e)}")
            # Return a simple fallback sample in case of errors
            dummy_tokens = torch.ones(100, dtype=torch.long) * self.tokenizer.pad_token_id
            return {
                "input_ids": dummy_tokens,
                "attention_mask": torch.zeros_like(dummy_tokens),
                "labels": torch.full_like(dummy_tokens, -100),
                "task": "tts_token",
                "sample_id": cut['id']
            }


def create_libriheavy_dataloaders(
    tokenizer: PreTrainedTokenizer,
    manifest_dir: str = "path/to/data/libriheavy/",
    manifest_file: str = "libriheavy_cuts_large.jsonl.gz",
    audio_base_path: str = "path/to/data/libriheavy/",
    dataset_type: str = "asr_token",  # Options: "asr_wave", "asr_token", "tts_token"
    max_seq_length: int = 2048,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 4,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    split_ratio: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for LibriHeavy datasets.
    
    Args:
        tokenizer: Tokenizer for encoding inputs and outputs
        manifest_dir: Directory containing LibriHeavy manifest files
        manifest_file: Name of the manifest file
        audio_base_path: Base path to prepend to audio file paths
        dataset_type: Type of dataset to create ("asr_wave", "asr_token", or "tts_token")
        max_seq_length: Maximum sequence length for inputs
        max_train_samples: Maximum number of training samples (None = use split_ratio)
        max_val_samples: Maximum number of validation samples (None = use split_ratio)
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for validation
        num_workers: Number of workers for dataloaders
        world_size: Number of processes in distributed training
        rank: Rank of the current process
        seed: Random seed for reproducibility
        split_ratio: Ratio to split between train and validation (0.9 = 90% train)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Choose dataset class based on dataset_type
    if dataset_type == "asr_wave":
        dataset_class = LibriHeavyASRWaveDataset
    elif dataset_type == "asr_token":
        dataset_class = LibriHeavyASRTokenDataset
    elif dataset_type == "tts_token":
        dataset_class = LibriHeavyTTSTokenDataset
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. " 
                         f"Must be one of 'asr_wave', 'asr_token', or 'tts_token'.")
    
    # Create training dataset
    train_dataset = dataset_class(
        tokenizer=tokenizer,
        manifest_dir=manifest_dir,
        manifest_file=manifest_file,
        audio_base_path=audio_base_path,
        max_seq_length=max_seq_length,
        max_samples=max_train_samples,
        split_ratio=split_ratio,
        is_train=True,
        seed=seed
    )
    
    # Create distributed sampler for training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
    else:
        train_sampler = None
    
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    
    return train_dataloader

if __name__ == "__main__":
    tokenizer_path = "path/to/model/MoST-initialized"
    # Load the custom tokenizer directly instead of using AutoTokenizer
    tokenizer = MoSTTokenizerFast.from_pretrained(tokenizer_path)
    for i in range(5):
        print(tokenizer.decode(100000 + i))
    print(tokenizer.begin_audio_token_id)
    print(tokenizer.end_audio_token_id)
    print(tokenizer.decode(tokenizer.begin_audio_token_id))
    print(tokenizer.decode(tokenizer.end_audio_token_id))
    
    # Define data directories
    asr_dirs = ["path/to/data/librispeech_asr_instructions", 
                "path/to/data/common_voice_asr_instructions"]
    tts_dirs = ["path/to/data/librispeech_tts_instructions", 
                "path/to/data/common_voice_tts_instructions"]
    
    # Define file patterns
    train_asr_pattern = "*train*.json"
    train_tts_pattern = "*train*.json"
    # Use separate patterns for dev and valid files
    val_asr_patterns = ["*dev*.json", "*valid*.json"]
    val_tts_patterns = ["*dev*.json", "*valid*.json"]
    

    # Alternative approach: create datasets directly
    train_dataset = MultimodalInstructionDataset(
        tokenizer=tokenizer, 
        asr_data_dirs=asr_dirs, 
        tts_data_dirs=tts_dirs,
        asr_file_patterns=train_asr_pattern,
        tts_file_patterns=train_tts_pattern
    )
    
    val_dataset = MultimodalInstructionDataset(
        tokenizer=tokenizer, 
        asr_data_dirs=asr_dirs, 
        tts_data_dirs=tts_dirs,
        asr_file_patterns=val_asr_patterns,
        tts_file_patterns=val_tts_patterns
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Print a sample to verify
    sample = train_dataset[0]
    print(f"Sample task: {sample['task']}")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Sample input: {tokenizer.decode(sample['input_ids'])}")
    print(f"Label values: {sample['labels'][:20]} ... (showing first 20)")
    print(f"Min/Max label values: {min(sample['labels'])}, {max(sample['labels'])}")

    # Filter out -100 values (ignored indices) before decoding
    valid_label_ids = [id for id in sample['labels'] if id != -100]
    print(f"Valid label count: {len(valid_label_ids)}")
    if valid_label_ids:
        print(f"Sample output: {tokenizer.decode(valid_label_ids)}")
    else:
        print("Sample output: [Empty - all labels are masked]")

    # Test LibriHeavy datasets
    print("\n=== Testing LibriHeavy Datasets ===")
    manifest_dir = "path/to/data/libriheavy/"
    audio_base_path = "path/to/data/libriheavy/"
    
    # Test ASR Token Dataset with a small sample size
    asr_token_dataset = LibriHeavyASRTokenDataset(
        tokenizer=tokenizer,
        manifest_dir=manifest_dir,
        audio_base_path=audio_base_path,
        max_samples=5  # Only load 5 samples for testing
    )
    
    print(f"\nASR Token Dataset size: {len(asr_token_dataset)}")
    for i in range(min(2, len(asr_token_dataset))):
        sample = asr_token_dataset[i]
        print(f"\nSample {i} - Task: {sample['task']}, ID: {sample['sample_id']}")
        print(f"Input tokens shape: {sample['input_ids'].shape}")
        print(f"Input text: {tokenizer.decode(sample['input_ids'])[:100]}...")
        
        # Filter out -100 values (ignored indices) before decoding
        valid_label_ids = [id for id in sample['labels'] if id != -100]
        print(f"Valid label count: {len(valid_label_ids)}")
        if valid_label_ids:
            print(f"Output text: {tokenizer.decode(valid_label_ids)[:100]}...")
    
    # Test TTS Token Dataset with a small sample size
    tts_token_dataset = LibriHeavyTTSTokenDataset(
        tokenizer=tokenizer,
        manifest_dir=manifest_dir,
        audio_base_path=audio_base_path,
        max_samples=5  # Only load 5 samples for testing
    )
    
    print(f"\nTTS Token Dataset size: {len(tts_token_dataset)}")
    for i in range(min(2, len(tts_token_dataset))):
        sample = tts_token_dataset[i]
        print(f"\nSample {i} - Task: {sample['task']}, ID: {sample['sample_id']}")
        print(f"Input tokens shape: {sample['input_ids'].shape}")
        print(f"Input text: {tokenizer.decode(sample['input_ids'])[:100]}...")
        
        # Filter out -100 values (ignored indices) before decoding
        valid_label_ids = [id for id in sample['labels'] if id != -100]
        print(f"Valid label count: {len(valid_label_ids)}")
        if valid_label_ids:
            print(f"Output contains audio tokens: {'<|begin_of_audio|>' in tokenizer.decode(valid_label_ids)}")
            print(f"Output preview: {tokenizer.decode(valid_label_ids)[:100]}...")