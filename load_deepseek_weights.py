import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from configuration_MoST import MoSTConfig, MoSTCConfig
from modeling_most import MoSTForCausalLM, MoSTCForCausalLM
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weight_mapping(config):
    """
    Define the mapping between DeepSeek model weights and MoST model weights.
    Returns a dictionary where keys are MoST model parameter names and values are DeepSeek model parameter names.
    If a parameter name is not present in the mapping, it will be initialized from scratch.

    Args:
        config: The MoST model configuration object.

    Returns:
        A dictionary where keys are MoST model parameter names and 
        values are corresponding DeepSeek model parameter names.
    """
    mapping = {}
    
    # Embeddings
    mapping["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    
    # Layers mapping - Use config attributes
    num_layers = config.num_hidden_layers
    n_routed_experts = getattr(config, 'n_routed_experts', None)
    first_k_dense = getattr(config, 'first_k_dense_replace', 0)
    moe_freq = getattr(config, 'moe_layer_freq', 0)
    has_moe = n_routed_experts is not None and moe_freq > 0

    logger.info(f"Generating weight map for {num_layers} layers.")
    if has_moe:
        logger.info(f"MoE config: first_k_dense={first_k_dense}, moe_freq={moe_freq}, num_experts={n_routed_experts}")
        
    for i in range(num_layers):
        # --- Attention Mapping (DeepSeek V2 Lite specific) ---
        # Assuming MoSTC model adopts this structure for weight loading compatibility
        mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        # Map DeepSeek's specific K/V projections to MoSTC (assuming compatibility)
        mapping[f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] = f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"
        mapping[f"model.layers.{i}.self_attn.kv_a_layernorm.weight"] = f"model.layers.{i}.self_attn.kv_a_layernorm.weight"
        mapping[f"model.layers.{i}.self_attn.kv_b_proj.weight"] = f"model.layers.{i}.self_attn.kv_b_proj.weight"
        # Standard O-proj
        mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        
        # --- Layer Norms (Standard) ---
        mapping[f"model.layers.{i}.input_layernorm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
        
        # --- MLP or MoE Mapping ---
        is_moe_layer = has_moe and i >= first_k_dense and i % moe_freq == 0
        
        if is_moe_layer:
            # This is an MoE layer
            logger.debug(f"Mapping MoE layer {i}")
            # Map the gate (router)
            mapping[f"model.layers.{i}.mlp.gate.weight"] = f"model.layers.{i}.mlp.gate.weight"
            # Map the experts (using standard proj names, assuming MoSTC matches DeepSeek's expert structure)
            for j in range(n_routed_experts):
                 # Deepseek V2 Lite expert naming uses gate/up/down_proj within each expert folder
                 mapping[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"
                 mapping[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"
                 mapping[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"
            # Map the shared experts
            mapping[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            mapping[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            mapping[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
        else:
            # This is a dense MLP layer
            logger.debug(f"Mapping Dense MLP layer {i}")
            mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
            mapping[f"model.layers.{i}.mlp.down_proj.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
            mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
            
    # Final norm and lm_head
    mapping["model.norm.weight"] = "model.norm.weight"
    mapping["lm_head.weight"] = "lm_head.weight"
    
    return mapping

def load_deepseek_weights(most_model, deepseek_model_path, save_path=None):
    """
    Load weights from a DeepSeek-V2 Lite model into a MoST model.
    
    Args:
        most_model: The MoST model to load weights into
        deepseek_model_path: Path to the DeepSeek model
        save_path: Optional path to save the initialized MoST model
    
    Returns:
        The MoST model with initialized weights
    """
    # Load DeepSeek model
    logger.info(f"Loading DeepSeek model from {deepseek_model_path}")
    deepseek_model = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Get DeepSeek model state dict
    deepseek_state_dict = deepseek_model.state_dict()
    
    # Get MoST model state dict
    most_state_dict = most_model.state_dict()
    
    # Get weight mapping
    mapping = get_weight_mapping(most_model.config)
    
    # Load weights from DeepSeek to MoST
    logger.info("Loading weights from DeepSeek to MoST")
    transferred_keys = []
    initialized_keys = []
    
    for most_key, most_param in most_state_dict.items():
        if most_key in mapping and mapping[most_key] in deepseek_state_dict:
            deepseek_param = deepseek_state_dict[mapping[most_key]]
            
            # Check if shapes match
            if most_param.shape == deepseek_param.shape:
                most_state_dict[most_key] = deepseek_param
                transferred_keys.append(most_key)
            else:
                logger.warning(f"Shape mismatch for {most_key}: MoST shape {most_param.shape} vs DeepSeek shape {deepseek_param.shape}")
                initialized_keys.append(most_key)
        else:
            # If the key is not in the mapping, it's a new parameter (like MoE weights)
            initialized_keys.append(most_key)
    
    # Load weights into MoST model
    missing_keys, unexpected_keys = most_model.load_state_dict(most_state_dict, strict=False)
    
    # Log statistics
    logger.info(f"Transferred {len(transferred_keys)} parameters from DeepSeek")
    logger.info(f"Initialized {len(initialized_keys)} parameters from scratch")
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")
    
    # Save the model if a save path is provided
    if save_path:
        logger.info(f"Saving initialized MoST model to {save_path}")
        
        # Calculate total model size to determine shard size for 4 shards
        total_size_bytes = sum(param.numel() * param.element_size() for param in most_model.parameters())
        # Add some buffer (10%) to account for additional metadata
        shard_size_bytes = (total_size_bytes // 4) + (total_size_bytes // 40)
        
        # Convert to string format with GB units that save_pretrained expects
        max_shard_size = "8GB"
        logger.info(f"Using max_shard_size of {max_shard_size} to create approximately 4 shards")
        
        most_model.save_pretrained(save_path, max_shard_size=max_shard_size, torch_dtype=torch.bfloat16)
    
    return most_model

def load_deepseek_hubert_weights(mostc_model: MoSTCForCausalLM, deepseek_model_path: str, save_path: Optional[str] = None):
    """
    Load weights from a DeepSeek-V2 Lite model into the text/transformer parts 
    of a MoSTCForCausalLM model.
    
    Assumes the HuBERT weights within the AudioWaveProcessor are loaded 
    during MoSTCForCausalLM initialization based on its config.
    
    Args:
        mostc_model: The MoSTCForCausalLM model to load weights into.
        deepseek_model_path: Path to the pre-trained DeepSeek model.
        save_path: Optional path to save the initialized MoSTC model.
    
    Returns:
        The MoSTC model with initialized weights.
    """
    # Load DeepSeek model
    logger.info(f"Loading DeepSeek base model from {deepseek_model_path}")
    deepseek_model = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        # local_files_only=True # Consider adding back if models are local
    )

    # --- Debug: Print Layer 0 Parameter Names ---
    # logger.info("--- DeepSeek Layer 5 Parameter Names ---")
    # for name, param in deepseek_model.named_parameters():
    #     if name.startswith("model.layers.5."):
    #         logger.info(f"Layer 5 Param: {name} | Shape: {param.shape}")
    # logger.info("---------------------------------------")
    # breakpoint()
    # --- End Debug ---
    
    # Get DeepSeek model state dict
    deepseek_state_dict = deepseek_model.state_dict()
    
    # Get MoST-C model state dict
    mostc_state_dict = mostc_model.state_dict()
    
    # Get weight mapping (should be the same for common components)
    mapping = get_weight_mapping(mostc_model.config)
    
    # Load weights from DeepSeek to MoST-C (text/transformer parts)
    logger.info("Loading weights from DeepSeek to MoST-C")
    transferred_deepseek_keys = []
    initialized_other_keys = []
    
    for mostc_key, mostc_param in mostc_state_dict.items():
        if mostc_key in mapping and mapping[mostc_key] in deepseek_state_dict:
            deepseek_param = deepseek_state_dict[mapping[mostc_key]]
            
            # Check if shapes match
            if mostc_param.shape == deepseek_param.shape:
                mostc_state_dict[mostc_key] = deepseek_param
                transferred_deepseek_keys.append(mostc_key)
            else:
                logger.warning(
                    f"Shape mismatch for {mostc_key}: MoST-C shape {mostc_param.shape} vs DeepSeek shape {deepseek_param.shape}. "
                    f"Keeping MoST-C initialization."
                )
                initialized_other_keys.append(mostc_key)
        else:
            # These keys are not in the mapping or not in DeepSeek.
            # They could be MoE-specific, audio-processor specific, or new params.
            # Assume they are correctly initialized by MoSTCForCausalLM constructor.
            if 'audio_wave_processor' in mostc_key:
                 # Log audio processor keys separately if desired
                 pass # logger.debug(f"Keeping initialized audio processor key: {mostc_key}")
            initialized_other_keys.append(mostc_key)
            
    # Load the modified state dict into MoST-C model
    # strict=False allows keys present in mostc_model but not loaded here (like audio processor)
    missing_keys, unexpected_keys = mostc_model.load_state_dict(mostc_state_dict, strict=False)
    
    # Log statistics
    logger.info(f"Transferred {len(transferred_deepseek_keys)} parameters from DeepSeek")
    logger.info(f"Kept {len(initialized_other_keys)} initialized parameters in MoST-C (incl audio processor, etc.)")
    
    # --- Sanity Check --- 
    # Identify which missing keys SHOULD have been loaded from DeepSeek based on the mapping
    deepseek_source_keys = set(mapping.values())
    unexpectedly_missing_deepseek_keys = []
    potentially_mostc_specific_missing_keys = []

    for key in missing_keys:
        # Check if the key corresponds to a parameter that should exist in DeepSeek according to our mapping
        # We check if the MoST-C key (which is the one reported as missing) exists in our mapping's *keys*
        if key in mapping and mapping[key] in deepseek_source_keys:
            unexpectedly_missing_deepseek_keys.append(key)
        elif 'audio_wave_processor' not in key: # Ignore audio processor keys if missing
             potentially_mostc_specific_missing_keys.append(key)

    if unexpectedly_missing_deepseek_keys:
        logger.warning(f"SANITY CHECK WARNING: {len(unexpectedly_missing_deepseek_keys)} keys expected from DeepSeek were MISSING in the loaded state_dict:")
        for key in unexpectedly_missing_deepseek_keys:
            logger.warning(f"  - {key} (Expected DeepSeek source: {mapping.get(key)})")
    else:
        logger.info("SANITY CHECK: All expected DeepSeek keys appear to be present or accounted for.")
        
    if potentially_mostc_specific_missing_keys:
        logger.info(f"INFO: {len(potentially_mostc_specific_missing_keys)} MoST-C specific keys (excluding audio) were missing (may be expected if not in mapping): {potentially_mostc_specific_missing_keys}")
    # --- End Sanity Check ---

    # Filter unexpected keys - DeepSeek keys not mapped or used in MoST-C
    # unexpected_keys contains keys from the provided state_dict (mostc_state_dict) that are not in the model's definition.
    # This shouldn't happen here as we start with mostc_model.state_dict(), unless the model definition changed.
    # Let's refine the original filtering to be more precise: filter keys from the *original* deepseek dict that weren't used.
    used_deepseek_keys = set(mapping[mk] for mk in transferred_deepseek_keys)
    unused_deepseek_keys = [k for k in deepseek_state_dict.keys() if k not in used_deepseek_keys]

    # Filter the list to avoid excessive logging, maybe show first 10-20?
    max_log_unused = 20
    logger.info(f"INFO: Found {len(unused_deepseek_keys)} DeepSeek keys that were not used in MoST-C mapping.")
    if unused_deepseek_keys:
        logger.info(f"  First {min(len(unused_deepseek_keys), max_log_unused)} unused DeepSeek keys: {unused_deepseek_keys[:max_log_unused]}")
    # logger.info(f"Missing keys in loaded state_dict (expected if MoST-C has extra params): {missing_keys}") # Covered by sanity check
    # logger.info(f"Unexpected keys from DeepSeek (potentially unused): {unexpected_keys}") # Covered by unused_deepseek_keys logging
    
    # Save the model if a save path is provided
    if save_path:
        logger.info(f"Saving initialized MoST-C model to {save_path}")
        
        # Use a reasonable default shard size or calculate dynamically
        total_size_bytes = sum(p.numel() * p.element_size() for p in mostc_model.parameters())
        # Estimate shard size for ~4 shards, add buffer
        # Adjust the divisor based on desired number of shards
        num_shards_approx = 4 
        shard_size_gb = (total_size_bytes / (1024**3)) / num_shards_approx
        # Use a practical shard size like 8GB or 10GB if calculation is too small/large
        max_shard_size = "8GB" 
        logger.info(f"Calculated total model size: {total_size_bytes / (1024**3):.2f} GB")
        logger.info(f"Using max_shard_size='{max_shard_size}' for saving.")
        
        mostc_model.save_pretrained(save_path, max_shard_size=max_shard_size, torch_dtype=torch.bfloat16)
        logger.info(f"MoST-C model saved to {save_path}")

    return mostc_model

def main():
    # --- Configuration ---
    # Set USE_MOST_C to True to test the MoST-C loading function
    USE_MOST_C = True 
    
    deepseek_model_path = "" 
    # If using MoST-C, ensure config_mostc.json exists and has HuBERT paths
    config_path = "config_mostc.json" if USE_MOST_C else "config.json" 
    # Define where to save the initialized model
    save_path = "" if USE_MOST_C else ""
    # --- End Configuration ---

    # Create MoST config and model
    logger.info(f"Loading MoST config from {config_path}")
    # Use MoSTConfig for both, as MoSTCConfig might not exist separately yet
    # The actual config values (like use_continuous_audio) determine behavior
    if USE_MOST_C:
        config = MoSTCConfig.from_pretrained(config_path) 
    else:
        config = MoSTConfig.from_pretrained(config_path) 
    
    # Create MoST model (either standard or C variant)
    if USE_MOST_C:
        logger.info("Creating MoST-C model with random initialization")
        # MoSTCForCausalLM will initialize AudioWaveProcessor based on config
        most_model = MoSTCForCausalLM(config)

        # --- Debug: Print Layer 0 Parameter Names ---
        # logger.info("--- MoST-C Layer 5 Parameter Names ---")
        # for name, param in most_model.named_parameters():
        #     if name.startswith("model.layers.5."):
        #         logger.info(f"Layer 5 Param: {name} | Shape: {param.shape}")
        # logger.info("---------------------------------------")
        # --- End Debug ---

        # Load DeepSeek weights into MoST-C
        most_model = load_deepseek_hubert_weights(
            most_model, 
            deepseek_model_path, 
            save_path=save_path
        )
    else:
        logger.info("Creating MoST model with random initialization")
        most_model = MoSTForCausalLM(config)
        # Load DeepSeek weights into standard MoST
        most_model = load_deepseek_weights(
            most_model, 
            deepseek_model_path, 
            save_path=save_path
        )
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 