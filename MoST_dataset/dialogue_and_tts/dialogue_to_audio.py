import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import json
import os
import random
from typing import List, Dict
import re
from datasets import load_dataset
import inflect
import torch.multiprocessing as mp

def log_to_file(message, file_path="terminal_print(dialogue_to_audio).txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")

# Speaker descriptions
SPEAKER_DESCRIPTIONS = {
    "Jon": "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
    "Lea": "Lea's voice is warm and measured, speaking at a moderate pace with clear articulation and minimal background noise.",
    "Gary": "Gary's voice is deep and authoritative, maintaining a steady rhythm with professional studio-quality recording.",
    "Jenna": "Jenna's voice is monotone yet slightly slow in delivery, with a very close recording that almost has no background noise.",
    "Mike": "Mike's voice is energetic but controlled, speaking at a brisk pace with pristine audio quality.",
    "Laura": "Laura's voice is gentle and precise, maintaining an even tempo with excellent recording clarity."
}

p = inflect.engine()

letter_to_word = {
    'A': 'ay',
    'B': 'bee',
    'C': 'see',
    'D': 'dee',
    'E': 'ee',
    'F': 'eff',
    'G': 'G',
    'H': 'aitch',
    'I': 'eye',
    'J': 'jay',
    'K': 'kay',
    'L': 'ell',
    'M': 'em',
    'N': 'en',
    'O': 'oh',
    'P': 'pee',
    'Q': 'cue',
    'R': 'ar',
    'S': 'ess',
    'T': 'tee',
    'U': 'you',
    'V': 'vee',
    'W': 'double you',
    'X': 'ex',
    'Y': 'why',
    'Z': 'zee'
}

class DialogueAudioConverter:
    def __init__(self, cache_dir: str, output_dir: str, device: str):
        self.device = device
        log_to_file(f"Using device: {self.device}")
        
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-large-v1",
            cache_dir=cache_dir
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-large-v1",
            cache_dir=cache_dir
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def split_text_into_chunks(self, text: str, max_words: int = 20) -> List[str]:  # Test Here
        """Split text into chunks of complete sentences with maximum word count."""
        sentences = re.split('(?<=[.!?\n])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count <= max_words:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def generate_audio(self, text: str, speaker_description: str) -> np.ndarray:
        """Generate audio for a given text and speaker description."""
        inputs = self.tokenizer(speaker_description, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        generation = self.model.generate(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt_inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt_inputs.attention_mask
        )
        
        audio = generation.cpu().numpy()
        # log_to_file(f"Generated audio shape before squeeze: {audio.shape}")
        # log_to_file(f"Generated audio dtype: {audio.dtype}")
        
        audio = audio.squeeze(axis=0)
        # log_to_file(f"Generated audio shape after squeeze: {audio.shape}")
        
        return audio

    def process_dialogue(self, dialogue: List[Dict], dialogue_id: int) -> Dict:
        """Process a complete dialogue and return metadata."""
        # Randomly assign two different speakers
        speakers = random.sample(list(SPEAKER_DESCRIPTIONS.items()), 3)
        speaker_map = {
            "user": speakers[0],
            "assistant": speakers[1]
        }
        
        dialogue_dir = os.path.join(self.output_dir, f"dialogue_{dialogue_id}")
        os.makedirs(dialogue_dir, exist_ok=True)
        
        metadata = {
            "dialogue_id": dialogue_id,
            "speakers": {role: speaker[0] for role, speaker in speaker_map.items()},
            "turns": []
        }

        
        for turn_idx, turn in enumerate(dialogue['messages']):
            # print(turn_idx, turn)
            role = turn["role"]
            content = turn["content"]

            if role == 'system':    # deal with system
                log_to_file(f'system skipped.')

                # Add turn metadata
                metadata["turns"].append({
                    "role": role,
                    "content": content
                })

                # Save turn metadata
                turn_metadata = {
                    "role": role,
                    "content": content
                }
                turn_info_path = os.path.join(dialogue_dir, f"turn_{turn_idx}_info.json")
                with open(turn_info_path, "w") as info_file:
                    json.dump(turn_metadata, info_file, indent=2)
                continue

            speaker_name, speaker_desc = speaker_map[role]

            # Process calculations and expressions
            content = self.process_expression(content)
            
            # Split content into chunks if necessary
            chunks = self.split_text_into_chunks(content)
            chunk_audios = []

            # Save turn metadata   ---- debug
            # debug_metadata = {
            #     "role": role,
            #     "speaker": speaker_name,
            #     "content": content
            # }
            # debug_info_path = os.path.join(dialogue_dir, f"debug_{turn_idx}_info.json")
            # with open(debug_info_path, "w") as info_file:
            #     json.dump(debug_metadata, info_file, indent=2)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Generate audio for chunk
                try:
                    audio = self.generate_audio(chunk, speaker_desc)
                except ValueError as e:
                    if "Token indices sequence length is longer than the specified maximum sequence length" in str(e):
                        metadata["turns"].append({
                            "role": role,
                            "speaker": speaker_name,
                            "content": content,
                            "exception": str(e)
                        })
                    return metadata

                
                # Save chunk audio
                chunk_filename = f"turn_{turn_idx}_chunk_{chunk_idx}.wav"
                chunk_path = os.path.join(dialogue_dir, chunk_filename)
                sf.write(chunk_path, audio, self.model.config.sampling_rate)
                chunk_audios.append(chunk_filename)
            
            # If there are multiple chunks, concatenate them
            if len(chunks) > 1:
                full_audio = np.concatenate([
                    sf.read(os.path.join(dialogue_dir, f))[0] 
                    for f in chunk_audios
                ])
                turn_filename = f"turn_{turn_idx}_full.wav"
                turn_path = os.path.join(dialogue_dir, turn_filename)
                sf.write(turn_path, full_audio, self.model.config.sampling_rate)

                log_to_file(f'data saved.')
            else:
                turn_filename = chunk_audios[0]
                log_to_file(f'Chunk<=1')
            
            
            # Add turn metadata
            metadata["turns"].append({
                "role": role,
                "speaker": speaker_name,
                "content": content,
                "audio_file": turn_filename,
                "chunk_files": chunk_audios
            })

            # Save turn metadata
            turn_metadata = {
                "role": role,
                "speaker": speaker_name,
                "content": content,
                "audio_file": turn_filename,
                "chunk_files": chunk_audios
            }
            turn_info_path = os.path.join(dialogue_dir, f"turn_{turn_idx}_info.json")
            with open(turn_info_path, "w") as info_file:
                json.dump(turn_metadata, info_file, indent=2)

        # Save all turns metadata
        dialogue_metadata_path = os.path.join(dialogue_dir, f"dialogue_{dialogue_id}_info.json")
        with open(dialogue_metadata_path, "w") as dialogue_file:
            json.dump(metadata, dialogue_file, indent=2)
        
        return metadata

    def process_dataset(self, dataset: List[List[Dict]]):
        """Process entire dataset and save metadata."""
        all_metadata = []
        
        for dialogue_id, dialogue in enumerate(dataset):

            log_to_file(f"Processing dialogue: {dialogue_id}")
            
            if self.dialogue_filter(dialogue):
                log_to_file(f'Processing dialogue: {dialogue_id} -> code is skipped.')
                continue

            metadata = self.process_dialogue(dialogue, dialogue_id)
            all_metadata.append(metadata)
        
        # Save complete metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f, indent=2)
    
    # Process unreadable words
    def process_expression(self, text: str):
        
        def replace_capital(match):
            return letter_to_word[match.group(0)]

        # Add space between number and letter
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        # Process number
        numbers = re.findall(r'\d+', text)
        for number in numbers:
            word = p.number_to_words(number)
            text = text.replace(number, word, 1)
        # Process calculation
        text = text.replace('*', 'times').replace('/', 'divided by')
        text = text.replace('+', 'plus').replace('-', 'minus')
        text = text.replace('=', 'equals')
        # Process bracket
        text = text.replace('\\(', '').replace('\\)', '')
        text = text.replace('\\[', '').replace('\\]', '')
        text = re.sub(r'([a-zA-Z])\(', r'\1 ', text)
        text = re.sub(r'([a-zA-Z0-9])\)', r'\1', text)
        text = text.replace('(', 'open parenthese').replace(')', 'close parenthese')
        text = text.replace('[', 'open bracket').replace(']', 'close bracket')
        text = text.replace(' .', '.').replace('\u2019', "'")

        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\b(?!a\b)([a-z])\b', lambda m: m.group(1).upper(), text)
        text = re.sub(r'(?<![A-Za-z])[A-Z](?![A-Za-z])', replace_capital, text)

        return text

    def dialogue_filter(self, dialogue):
        code_keywords = r'\b(def|C\+\+|java|javascript|#include|const|int)\b'
        for turn_idx, turn in enumerate(dialogue['messages']):
            role = turn["role"]
            content = turn["content"]
            if role == 'assistant' and re.search(code_keywords, content, re.IGNORECASE):
                match = re.search(code_keywords, content, re.IGNORECASE)
                log_to_file(f"Match content: {match.group()}, Match position: {match.span()}")
                return True
        return False
        
def worker_process(rank: int, world_size: int, proc_on_gpu:int, dataset, output_dir: str, cache_dir: str):
    # Set process GPU
    torch.cuda.set_device(rank//proc_on_gpu)
    device = f"cuda:{rank//proc_on_gpu}"
    
    # initialize converter
    converter = DialogueAudioConverter(cache_dir, output_dir, device)
    
    # Process data
    for idx in range(rank, len(dataset), world_size):
        if converter.dialogue_filter(dataset[idx]):
            log_to_file(f'Dialogue {idx} is skipped.')
            continue
        converter.process_dialogue(dataset[idx], idx)
        log_to_file(f"Processed dialogue {idx} on GPU {rank//proc_on_gpu}")

if __name__ == "__main__":
    # Configuration
    model_cache_dir = "path/to/ParlerTTS"
    output_dir = "path/to/smoltalk_audio"
    dataset_cache_dir = "path/to/smoltalk"
    dataset_name = "HuggingFaceTB/smoltalk"

    # Load SmolTalk dataset
    dataset = load_dataset(dataset_name, 'all',split="train[:10000]", cache_dir=dataset_cache_dir)
    log_to_file(f'dataset loaded')
    
    # Initialize converter
    # converter = DialogueAudioConverter(model_cache_dir, output_dir)
    
    # Process dataset
    # converter.process_dataset(dataset)

    # Get available GPU
    num_gpus = torch.cuda.device_count()
    log_to_file(f"Starting processing with {num_gpus} GPUs")
    proc_on_gpu = 2
    
    # Initialize multiprocess
    mp.spawn(worker_process,
             args=(num_gpus, proc_on_gpu, dataset, output_dir, model_cache_dir),
             nprocs=num_gpus * proc_on_gpu,
             join=True)