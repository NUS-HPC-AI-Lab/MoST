#!/usr/bin/env python
# coding=utf-8

"""
Example script demonstrating how to use Modality-Aware Mixture of Experts (MAMoE)
for the MoST (Mixture of Speech and Text) model.
"""

import torch
from transformers import AutoTokenizer
from configuration_MoST import MoSTConfig
from modeling_most import MoSTForCausalLM

def main():
    # Initialize configuration with MAMoE routing
    config = MoSTConfig(
        # Model architecture parameters
        vocab_size=102400,
        hidden_size=1024,  # Smaller for example
        num_hidden_layers=12,  # Smaller for example
        num_attention_heads=16,  # Smaller for example
        
        # Standard MoE parameters
        n_shared_experts=1,  # One shared expert (accessible to all modalities)
        n_routed_experts=8,  # 8 total routed experts
        num_experts_per_tok=2,  # Each token selects 2 experts
        
        # MAMoE specific parameters
        use_modality_aware_routing=True,  # Enable modality-aware routing
        text_expert_indices=[0, 1, 2, 3],  # First 4 experts assigned to text
        audio_expert_indices=[4, 5, 6, 7],  # Last 4 experts assigned to audio
    )
    
    # Initialize model
    model = MoSTForCausalLM(config)
    
    # Mock input data - create sample input with both text and audio tokens
    # Text tokens: 0-100001
    # Audio tokens: 100002-100503
    batch_size = 2
    seq_len = 10
    
    # Create a sample input with both text and audio tokens
    input_ids = torch.randint(0, 100001, (batch_size, seq_len//2))  # First half: text tokens
    audio_ids = torch.randint(100002, 100503, (batch_size, seq_len//2))  # Second half: audio tokens
    
    # Concatenate to create mixed modality input
    input_ids = torch.cat([input_ids, audio_ids], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Model forward pass
    print("Running model forward pass with MAMoE routing...")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Print output shape
    print(f"Output logits shape: {outputs.logits.shape}")
    
    # Example of how modality marks are determined internally
    print("\nDemonstrating manual modality mark calculation:")
    modality_marks = torch.zeros_like(input_ids)
    modality_marks[(input_ids >= 100002) & (input_ids <= 100503)] = 1  # Mark audio tokens
    modality_marks[input_ids > 100503] = 2  # Mark image tokens (not used in this example)
    
    # Print input IDs and corresponding modality marks
    for i in range(batch_size):
        print(f"\nSample {i+1}:")
        for j in range(seq_len):
            modality = "Text" if modality_marks[i, j] == 0 else "Audio" if modality_marks[i, j] == 1 else "Image"
            print(f"  Token {j+1}: ID = {input_ids[i, j].item()}, Modality = {modality}")
    
    print("\nMAMoE routing will ensure:")
    print("- Text tokens (ID < 100002) are only routed to text experts (indices 0-3)")
    print("- Audio tokens (ID 100002-100503) are only routed to audio experts (indices 4-7)")
    print("- All tokens can access the shared expert")

if __name__ == "__main__":
    main() 