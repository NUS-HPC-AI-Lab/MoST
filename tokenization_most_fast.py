from typing import List, Optional, Union, Dict
from transformers.models.llama import LlamaTokenizerFast
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import AddedToken
import torch
import os
from huggingface_hub import hf_hub_download

HUBERT_TOKENS = [f"[Hu{i}]" for i in range(500)]
AUDIO_SPECIAL_TOKENS = {
    "begin_audio_token": "<｜begin▁of▁audio｜>",
    "end_audio_token": "<｜end▁of▁audio｜>",
    "begin_audio_wave_token": "<｜begin▁of▁audio_wave｜>",
    "end_audio_wave_token": "<｜end▁of▁audio_wave｜>"
}

class MoSTTokenizerFast(LlamaTokenizerFast):
    """
    MoST tokenizer extending DeepSeek's tokenizer (which is based on LlamaTokenizerFast) with additional support for audio tokens.
    This includes 500 HuBERT tokens and special audio markers.
    """
    SPECIAL_TOKENS_ATTRIBUTES = LlamaTokenizerFast.SPECIAL_TOKENS_ATTRIBUTES + ["begin_audio_token", "end_audio_token", "begin_audio_wave_token", "end_audio_wave_token"]
    
    def __init__(self, *args, **kwargs):
        # Initialize special token attributes
        self._begin_audio_token = AUDIO_SPECIAL_TOKENS["begin_audio_token"]
        self._end_audio_token = AUDIO_SPECIAL_TOKENS["end_audio_token"]
        self._begin_audio_wave_token = AUDIO_SPECIAL_TOKENS["begin_audio_wave_token"]
        self._end_audio_wave_token = AUDIO_SPECIAL_TOKENS["end_audio_wave_token"]
        
        # Initialize the base tokenizer
        super().__init__(*args, **kwargs)
        
        # Add HuBERT tokens
        self.add_tokens(HUBERT_TOKENS)
        
        # Add special audio tokens
        special_tokens_dict = {
            "begin_audio_token": AddedToken(AUDIO_SPECIAL_TOKENS["begin_audio_token"], 
                                          lstrip=False, rstrip=False, normalized=True, single_word=False),
            "end_audio_token": AddedToken(AUDIO_SPECIAL_TOKENS["end_audio_token"], 
                                        lstrip=False, rstrip=False, normalized=True, single_word=False),
            "begin_audio_wave_token": AddedToken(AUDIO_SPECIAL_TOKENS["begin_audio_wave_token"],
                                               lstrip=False, rstrip=False, normalized=True, single_word=False),
            "end_audio_wave_token": AddedToken(AUDIO_SPECIAL_TOKENS["end_audio_wave_token"],
                                             lstrip=False, rstrip=False, normalized=True, single_word=False)
        }
        self.add_special_tokens(special_tokens_dict)
        
        # Cache the token IDs for quick access
        self._hubert_token_ids = {token: self.convert_tokens_to_ids(token) for token in HUBERT_TOKENS}
        self._begin_audio_token_id = self.convert_tokens_to_ids(AUDIO_SPECIAL_TOKENS["begin_audio_token"])
        self._end_audio_token_id = self.convert_tokens_to_ids(AUDIO_SPECIAL_TOKENS["end_audio_token"])
        self._begin_audio_wave_token_id = self.convert_tokens_to_ids(AUDIO_SPECIAL_TOKENS["begin_audio_wave_token"])
        self._end_audio_wave_token_id = self.convert_tokens_to_ids(AUDIO_SPECIAL_TOKENS["end_audio_wave_token"])

    @property
    def hubert_token_ids(self) -> Dict[str, int]:
        """Get the mapping of HuBERT tokens to their IDs."""
        return self._hubert_token_ids

    @property
    def begin_audio_token_id(self) -> int:
        """Get the ID of the begin audio token."""
        return self._begin_audio_token_id

    @property
    def end_audio_token_id(self) -> int:
        """Get the ID of the end audio token."""
        return self._end_audio_token_id

    @property
    def begin_audio_wave_token_id(self) -> int:
        """Get the ID of the begin audio wave token."""
        return self._begin_audio_wave_token_id

    @property
    def end_audio_wave_token_id(self) -> int:
        """Get the ID of the end audio wave token."""
        return self._end_audio_wave_token_id

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens.
        Handles both text and audio tokens.
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            token = self._tokenizer.id_to_token(index)
            tokens.append(token if token is not None else "")
        return tokens
    
    def _convert_id_to_token(self, index: int) -> Optional[str]:
        """Convert a single token id to its string representation."""
        token = self._tokenizer.id_to_token(int(index))
        return token if token is not None else ""

    def encode_audio_sequence(self, hubert_indices: List[int], add_special_tokens: bool = True) -> List[int]:
        """
        Encode a sequence of HuBERT indices into token IDs, optionally adding audio special tokens.
        
        Args:
            hubert_indices: List of HuBERT indices (0-499)
            add_special_tokens: Whether to add begin/end audio tokens
            
        Returns:
            List of token IDs
        """
        if not all(0 <= idx < 500 for idx in hubert_indices):
            raise ValueError("HuBERT indices must be between 0 and 499")
            
        # Convert HuBERT indices to tokens
        hubert_tokens = [f"[Hu{idx}]" for idx in hubert_indices]
        token_ids = [self._hubert_token_ids[token] for token in hubert_tokens]
        
        if add_special_tokens:
            token_ids = [self._begin_audio_token_id] + token_ids + [self._end_audio_token_id]
            
        return token_ids
        

if __name__ == "__main__":
    # Test tokenizer functionality
    try:
        # Try loading from saved path first
        tokenizer = MoSTTokenizerFast.from_pretrained("/path/to/your/initialized/MoST-initialized")
        print("Loaded MoST tokenizer from saved path")
    except Exception as e:
        print(f"Creating new MoST tokenizer: {e}")
        tokenizer = MoSTTokenizerFast()

    # Test the new audio wave special tokens
    print("\nTesting audio wave special tokens:")
    print(f"Begin audio wave token ID: {tokenizer.begin_audio_wave_token_id}")
    print(f"End audio wave token ID: {tokenizer.end_audio_wave_token_id}")
    print(f"Decoded begin token: {tokenizer.decode([tokenizer.begin_audio_wave_token_id])}")
    print(f"Decoded end token: {tokenizer.decode([tokenizer.end_audio_wave_token_id])}")

    # Save the updated tokenizer
    save_path = "/path/to/your/initialized/MoST-initialized"
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved updated tokenizer to {save_path}")

    # Test loading the saved tokenizer
    loaded_tokenizer = MoSTTokenizerFast.from_pretrained(save_path)
    print("\nVerifying loaded tokenizer:")
    print(f"Begin audio wave token matches: {loaded_tokenizer.decode([loaded_tokenizer.begin_audio_wave_token_id]) == '<｜begin▁of▁audio_wave｜>'}")
    print(f"End audio wave token matches: {loaded_tokenizer.decode([loaded_tokenizer.end_audio_wave_token_id]) == '<｜end▁of▁audio_wave｜>'}")

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    
    # Test text encoding/decoding
    text = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"
    encoded = tokenizer(text)
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"\nText encoding/decoding test:")
    print(f"Original: {text}")
    print(f"Decoded: {decoded}")
    
    # # Test audio token encoding/decoding
    # hubert_indices = [0, 1, 2, 3, 4]
    # audio_tokens = tokenizer.encode_audio_sequence(hubert_indices)
    # decoded_audio = tokenizer.decode(audio_tokens)
    # print(f"\nAudio encoding/decoding test:")
    # print(f"HuBERT indices: {hubert_indices}")
    # print(f"Decoded: {decoded_audio}")
    
    # # Test mixed text and audio
    # mixed_tokens = tokenizer(text)["input_ids"] + tokenizer.encode_audio_sequence(hubert_indices)
    # decoded_mixed = tokenizer.decode(mixed_tokens)
    # print(f"\nMixed text/audio test:")
    # print(f"Decoded: {decoded_mixed}")

    # # Test vocabulary boundary
    # print("\nChecking vocabulary boundary(text):")
    # for i in range(99995, 100005, 1):
    #     decoded = tokenizer.decode([i])
    #     if decoded.strip():  # Only print non-empty decoded tokens
    #         print(f"Token {i}: '{decoded}'")

    print("\nChecking vocabulary boundary(audio):")
    for i in range(100500, 100510, 1):
        decoded = tokenizer.decode([i])
        if decoded.strip():  # Only print non-empty decoded tokens
            print(f"Token {i}: '{decoded}'")
