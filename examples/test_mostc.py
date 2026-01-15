#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import numpy as np
from transformers import set_seed
from configuration_MoST import MoSTConfig
from modeling_most import MoSTCForCausalLM, MoSTForCausalLM, AudioWaveProcessor
import torch.nn.functional as F # Need F for loss calculation

def parse_args():
    parser = argparse.ArgumentParser(description="Test script for MoST-C model")
    parser.add_argument(
        "--most_model_path",
        type=str,
        default=None,
        help="Path to MoST model weights",
    )
    parser.add_argument(
        "--hubert_model_path",
        type=str,
        required=True,
        help="Path to HuBERT model checkpoint",
    )
    parser.add_argument(
        "--hubert_ckpt_type",
        type=str,
        default="converted",
        choices=["fairseq", "converted"],
        help="HuBERT checkpoint type (fairseq or converted)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--hubert_hidden_size",
        type=int,
        default=768,
        help="Hidden size of the HuBERT model",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="Number of experts in MoE layers",
    )
    parser.add_argument(
        "--use_modality_aware_routing",
        action="store_true",
        help="Whether to use modality-aware routing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="Directory to save test outputs",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    return parser.parse_args()


def test_mostc_initialization(args):
    """Test MoST-C model initialization"""
    print("\n=== Testing MoST-C Initialization ===")
    
    # Create a basic configuration for testing
    config = MoSTConfig(
        vocab_size=32000,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        
        # MoE parameters
        moe_intermediate_size=args.hidden_size * 4,
        num_experts=args.num_experts,
        num_experts_per_token=2,
        router_aux_loss_coef=0.01,
        
        # MAMoE parameters (used only if use_modality_aware_routing is True)
        use_modality_aware_routing=args.use_modality_aware_routing,
        text_expert_indices=[0, 1, 2, 3],
        audio_expert_indices=[4, 5, 6, 7],
        
        # MoST-C specific parameters
        use_continuous_audio=True,
        hubert_model_path=args.hubert_model_path,
        hubert_ckpt_type=args.hubert_ckpt_type,
        hubert_hidden_size=args.hubert_hidden_size,
        begin_audio_wave_token_id=100504,
        end_audio_wave_token_id=100505,
    )
    
    # Initialize the model
    model = MoSTCForCausalLM(config)
    
    # Check that key components are initialized correctly
    print(f"Model initialized successfully: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Base model is MoSTCModel: {model.model.__class__.__name__ == 'MoSTCModel'}")
    print(f"Has AudioWaveProcessor: {hasattr(model.model, 'audio_wave_processor')}")
    print(f"HuBERT parameters frozen: {not any(p.requires_grad for p in model.model.audio_wave_processor.hubert_model.parameters())}")
    
    return model, config


def analyze_model_differences(most_model, mostc_model):
    """Analyze the structural differences between MoST and MoST-C models"""
    print("\n=== Analyzing Model Structural Differences ===")
    
    # Get state dictionaries
    most_dict = most_model.state_dict()
    mostc_dict = mostc_model.state_dict()
    
    # Count parameters by prefix to identify where the differences are
    most_prefixes = {}
    mostc_prefixes = {}
    
    for key in most_dict:
        prefix = key.split('.')[0] if '.' in key else key
        most_prefixes[prefix] = most_prefixes.get(prefix, 0) + 1
    
    for key in mostc_dict:
        prefix = key.split('.')[0] if '.' in key else key
        mostc_prefixes[prefix] = mostc_prefixes.get(prefix, 0) + 1
    
    # Print top-level parameter counts
    print("\nTop-level parameter counts:")
    all_prefixes = sorted(set(list(most_prefixes.keys()) + list(mostc_prefixes.keys())))
    
    for prefix in all_prefixes:
        most_count = most_prefixes.get(prefix, 0)
        mostc_count = mostc_prefixes.get(prefix, 0)
        print(f"{prefix}: MoST: {most_count}, MoST-C: {mostc_count}, Diff: {mostc_count - most_count}")
    
    # Analyze model components
    print("\nModel component analysis:")
    print(f"MoST model: {most_model.__class__.__name__}")
    print(f"MoST-C model: {mostc_model.__class__.__name__}")
    print(f"MoST base model: {most_model.model.__class__.__name__}")
    print(f"MoST-C base model: {mostc_model.model.__class__.__name__}")
    
    # Check specific attributes
    most_attrs = dir(most_model.model)
    mostc_attrs = dir(mostc_model.model)
    
    most_only = set(most_attrs) - set(mostc_attrs)
    mostc_only = set(mostc_attrs) - set(most_attrs)
    
    if most_only:
        print("\nAttributes only in MoST model:", sorted(most_only))
    if mostc_only:
        print("\nAttributes only in MoST-C model:", sorted(mostc_only))
    
    # Check layer structure
    print("\nLayer structure:")
    try:
        print(f"MoST layers: {len(most_model.model.layers)}")
        print(f"MoST-C layers: {len(mostc_model.model.layers)}")
        
        # Check the structure of the first layer
        print("\nFirst layer comparison:")
        most_layer_keys = {k: v.shape for k, v in most_model.model.layers[0].state_dict().items()}
        mostc_layer_keys = {k: v.shape for k, v in mostc_model.model.layers[0].state_dict().items()}
        
        most_layer_only = {k: most_layer_keys[k] for k in set(most_layer_keys) - set(mostc_layer_keys)}
        mostc_layer_only = {k: mostc_layer_keys[k] for k in set(mostc_layer_keys) - set(most_layer_keys)}
        
        if most_layer_only:
            print("Parameters only in MoST first layer:", most_layer_only)
        if mostc_layer_only:
            print("Parameters only in MoST-C first layer:", mostc_layer_only)
    except Exception as e:
        print(f"Error comparing layers: {e}")

    print("Total parameters in MoST:", sum(p.numel() for p in most_model.parameters()))
    print("Total parameters in MoST-C:", sum(p.numel() for p in mostc_model.parameters()))
    
    return most_dict, mostc_dict


def test_mostc_weight_loading(args, config=None):
    """Test loading weights from a pre-trained MoST model to MoST-C"""
    print("\n=== Testing MoST-C Weight Loading ===")
    
    if args.most_model_path is None:
        print("Skipping weight loading test (no model path provided)")
        return None
    
    if config is None:
        # Create config if not provided
        config = MoSTConfig.from_pretrained(args.most_model_path)
        # Add MoST-C specific parameters
        config.use_continuous_audio = True
        config.hubert_model_path = args.hubert_model_path
        config.hubert_ckpt_type = args.hubert_ckpt_type
        config.hubert_hidden_size = args.hubert_hidden_size
        config.begin_audio_wave_token_id = 32001
        config.end_audio_wave_token_id = 32002
    
    # First load the original MoST model
    print(f"Loading original MoST model from {args.most_model_path}")
    most_model = MoSTForCausalLM.from_pretrained(args.most_model_path)
    
    # Now initialize a MoST-C model
    print("Initializing MoST-C model with the same configuration")
    mostc_model = MoSTCForCausalLM(config)
    
    # Analyze model differences to understand structural differences
    most_dict, mostc_dict = analyze_model_differences(most_model, mostc_model)
    
    # Copy weights from MoST to MoST-C for shared parameters
    print("\nCopying weights from MoST to MoST-C for shared parameters")
    
    # Print summary of model architectures
    print(f"MoST model has {len(most_dict)} parameters")
    print(f"MoST-C model has {len(mostc_dict)} parameters")
    
    # Critical components that should match 
    critical_prefixes = [
        'lm_head', 
        'model.embed_tokens',
        'model.layers', 
        'model.norm'
    ]
    
    # Count shared parameters by prefix
    shared_by_prefix = {prefix: 0 for prefix in critical_prefixes}
    missing_by_prefix = {prefix: 0 for prefix in critical_prefixes}
    
    # Count shared parameters
    shared_keys = 0
    shared_key_names = []
    
    # Try to match as many parameters as possible using normalized key names
    # Due to the different model class structures, we need to normalize the keys
    most_keys_normalized = {}
    for key in most_dict:
        # For both models, standardize layer references
        normalized_key = key.replace("model.model.", "model.")
        most_keys_normalized[normalized_key] = key
    
    # Now try to match using normalized keys
    for key in mostc_dict:
        normalized_key = key.replace("model.model.", "model.")
        
        if normalized_key in most_keys_normalized:
            # Found matching key via normalization
            most_key = most_keys_normalized[normalized_key]
            if mostc_dict[key].shape == most_dict[most_key].shape:
                mostc_dict[key] = most_dict[most_key]
                shared_keys += 1
                shared_key_names.append((key, most_key))
                
                # Update prefix counters
                for prefix in critical_prefixes:
                    if key.startswith(prefix):
                        shared_by_prefix[prefix] += 1
            else:
                print(f"Shape mismatch for normalized key: {key} <-> {most_key}, " 
                      f"MoST shape: {most_dict[most_key].shape}, "
                      f"MoST-C shape: {mostc_dict[key].shape}")
        # Original direct matching (as before)
        elif key in most_dict and mostc_dict[key].shape == most_dict[key].shape:
            mostc_dict[key] = most_dict[key]
            shared_keys += 1
            shared_key_names.append((key, key))
            
            # Update prefix counters
            for prefix in critical_prefixes:
                if key.startswith(prefix):
                    shared_by_prefix[prefix] += 1
        elif key in most_dict and mostc_dict[key].shape != most_dict[key].shape:
            print(f"Shape mismatch for key: {key}, " 
                  f"MoST shape: {most_dict[key].shape}, "
                  f"MoST-C shape: {mostc_dict[key].shape}")
        else:
            # Count keys that should be in MoST but aren't matching
            for prefix in critical_prefixes:
                if key.startswith(prefix):
                    missing_by_prefix[prefix] += 1
    
    print(f"\nShared key prefixes statistics:")
    for prefix in critical_prefixes:
        print(f"{prefix}: {shared_by_prefix[prefix]} shared, {missing_by_prefix[prefix]} missing in MoST")
        
    print(f"\nTotal: Successfully loaded {shared_keys} shared parameters")
    
    if shared_keys <= 10:  # Only print all shared keys if there are very few
        print("Shared keys (MoST-C -> MoST):")
        for mostc_key, most_key in shared_key_names:
            print(f"  {mostc_key} <- {most_key}")
    
    # Load the updated state dict
    mostc_model.load_state_dict(mostc_dict)
    
    return mostc_model, config


def test_modality_mask_handling(model, config):
    """Prepare inputs including dummy audio for testing."""
    print("\n=== Preparing Test Inputs (including Audio) ===")
    
    batch_size = 2
    seq_length = 32 # Increased length slightly
    audio_length = 16000 * 2 # 2 seconds of dummy audio per segment
    
    # Create input IDs with audio markers
    # Use vocab_size-10 to avoid special tokens potentially
    input_ids = torch.randint(100, config.vocab_size - 10, (batch_size, seq_length), dtype=torch.long)
    
    # --- Insert ONE audio segment per batch item ---
    # Ensure begin/end IDs are valid
    # if config.begin_audio_wave_token_id >= config.vocab_size or config.end_audio_wave_token_id >= config.vocab_size:
    #      raise ValueError("begin/end_audio_wave_token_id must be less than vocab_size")

    begin_pos = seq_length // 4
    # Ensure end_pos is after begin_pos and within bounds
    end_pos = begin_pos + 1 # Place markers adjacent for simplicity, content between them is ignored
    if end_pos >= seq_length:
        end_pos = seq_length - 1
        begin_pos = end_pos - 1

    # Assign markers
    input_ids[:, begin_pos] = config.begin_audio_wave_token_id
    input_ids[:, end_pos] = config.end_audio_wave_token_id

    # Create corresponding dummy audio waveforms: [total_audio_segments_in_batch, audio_len]
    # Since we have 1 segment per batch item, total_audio_segments = batch_size
    audio_wavs = torch.randn(batch_size, audio_length)

    # Create attention mask (all ones for this test)
    attention_mask = torch.ones_like(input_ids)

    # Create dummy audio attention mask (all ones as audio isn't padded)
    audio_attention_mask = torch.ones_like(audio_wavs)

    print("Input IDs shape:", input_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)
    print("Audio Waveforms shape:", audio_wavs.shape)
    print("Audio Attention Mask shape:", audio_attention_mask.shape)

    # Sample from batch 0
    print("\nSample from batch 0:")
    print("Input IDs:", input_ids[0].tolist())
    print(f"Begin audio token at position {begin_pos}, End audio token at position {end_pos}")

    # Return all necessary inputs
    return input_ids, attention_mask, audio_wavs, audio_attention_mask


def test_forward_backward(model, input_ids, attention_mask, audio_wavs, audio_attention_mask):
    """Test forward and backward pass with the MoST-C model using dummy loss."""
    print("\n=== Testing Forward and Backward Pass ===")

    # Set the model to training mode
    model.train()
    print(f"Input IDs Shape: {input_ids.shape}")

    # --- Forward pass WITHOUT labels ---
    print("Running forward pass...")
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_wavs=audio_wavs,
            audio_attention_mask=audio_attention_mask,
            output_hidden_states=True # Request hidden states for shape check
        )
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        raise

    # --- Check Output Shapes ---
    print("\nOutput shapes:")
    if hasattr(outputs, "logits"):
        print(f"Logits shape: {outputs.logits.shape}")
        # Expected shape: [batch_size, processed_seq_len, vocab_size]
        # Note: processed_seq_len will be different from input_ids.shape[1]
        processed_seq_len = outputs.logits.shape[1]
        print(f"Original sequence length: {input_ids.shape[1]}, Processed sequence length: {processed_seq_len}")
    else:
        print("Logits not found in outputs.")
        return None # Cannot proceed with backward pass

    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
         # hidden_states is a tuple of tensors, one per layer + embeddings
         print(f"Number of hidden states returned: {len(outputs.hidden_states)}")
         print(f"Shape of last hidden state: {outputs.hidden_states[-1].shape}")
         # Should match logits shape except for the last dimension
         if outputs.hidden_states[-1].shape[:2] != outputs.logits.shape[:2]:
              print(f"WARNING: Last hidden state shape {outputs.hidden_states[-1].shape[:2]} doesn't match logits shape {outputs.logits.shape[:2]}!")
    else:
         print("Hidden states not found in outputs.")


    # --- Backward pass with DUMMY loss ---
    # Create a simple dummy loss based on the logits to test gradient flow
    # For example, sum of squares of logits. Use float() to avoid potential type issues.
    dummy_loss = (outputs.logits.float() ** 2).mean()
    print(f"\nCalculated dummy loss: {dummy_loss.item()}")

    print("Running backward pass...")
    try:
        dummy_loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass FAILED: {e}")
        raise

    # Check if gradients are computed for trainable parameters
    grad_norm = 0.0
    trainable_params = 0
    non_trainable_params = 0
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += 1
            if param.grad is not None:
                param_grad_norm = torch.norm(param.grad.float()).item()
                grad_norm += param_grad_norm ** 2
                if param_grad_norm == 0:
                    zero_grad_params.append(name)
            else:
                # This case (requires_grad=True, grad=None) should generally not happen after backward
                # unless the parameter was not involved in the computation graph leading to the loss.
                print(f"WARNING: Trainable parameter '{name}' has no gradient.")
                zero_grad_params.append(name + " (None grad)")
        else:
            non_trainable_params += 1

    grad_norm = grad_norm ** 0.5
    print(f"\nGradient check:")
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total non-trainable parameters (incl. frozen Hubert): {non_trainable_params}")
    print(f"Total gradient norm (trainable params): {grad_norm}")
    if grad_norm == 0 and trainable_params > 0:
         print("WARNING: Gradient norm is zero. Check model connections or loss function.")
    if len(zero_grad_params) > 0:
         print(f"Parameters with zero or None gradient ({len(zero_grad_params)}):")
         # Print only a few if many
         print(zero_grad_params[:min(len(zero_grad_params), 10)])


    # Check that HuBERT params remain frozen (gradient should be None)
    hubert_grad_count = 0
    for name, param in model.model.audio_wave_processor.hubert_model.named_parameters():
        if param.grad is not None:
            hubert_grad_count += 1
            # print(f"WARNING: Frozen HuBERT parameter '{name}' has a gradient!")

    print(f"\nNumber of HuBERT parameters with gradients: {hubert_grad_count} (should be 0)")

    # Return model outputs for potential further inspection
    return outputs


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test model initialization
    model, config = test_mostc_initialization(args)
    
    print(f"Model Config: {config}")

    # Test weight loading if path is provided
    # if args.most_model_path:
    #     model, config = test_mostc_weight_loading(args, None)
    # else:
    #     raise ValueError("No model path provided")
    
    # Prepare inputs for forward/backward test
    input_ids, attention_mask, audio_wavs, audio_attention_mask = test_modality_mask_handling(model, config)

    # Test forward and backward using prepared inputs
    outputs = test_forward_backward(model, input_ids, attention_mask, audio_wavs, audio_attention_mask)

    if outputs:
        print("\n=== Test Script Completed Successfully ===")
    else:
        print("\n=== Test Script Completed with Errors (Forward/Backward failed) ===")


if __name__ == "__main__":
    main() 