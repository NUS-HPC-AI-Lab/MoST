#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert HuBERT model from fairseq to a simpler format")
    parser.add_argument(
        "--fairseq_path",
        type=str,
        required=True,
        help="Path to fairseq checkpoint file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for converted model",
    )
    return parser.parse_args()

def convert_hubert_model(fairseq_path, output_path):
    """
    Convert a HuBERT model from fairseq format to a simpler format 
    that can be used directly by MoST-C.
    """
    logger.info(f"Loading fairseq model from {fairseq_path}")
    checkpoint = torch.load(fairseq_path, map_location="cpu")
    logger.info(f"Checkpoint Keys: {checkpoint.keys()}")
    
    # Extract configuration from checkpoint
    if "model_cfg" not in checkpoint or "task_cfg" not in checkpoint:
        logger.warning("Using legacy fairseq checkpoint format")
        model_args = task_args = checkpoint["args"]
    else:
        model_args = checkpoint["model_cfg"]
        task_args = checkpoint["task_cfg"]
    
    # Extract model weights
    model_weights = checkpoint["model_weight"]
    
    # Extract dictionary symbols if available
    if "dictionaries_symbols" in checkpoint:
        dict_symbols = checkpoint["dictionaries_symbols"]
    else:
        dict_symbols = None
        logger.warning("Dictionary symbols not found in checkpoint, setting to None")
    
    # Create a new state dict with only the necessary components
    converted_state = {
        "model_cfg": vars(model_args) if hasattr(model_args, "__dict__") else model_args,
        "task_cfg": vars(task_args) if hasattr(task_args, "__dict__") else task_args,
        "model_weight": model_weights,
        "dictionaries_symbols": dict_symbols,
    }
    
    # Save the converted model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving converted model to {output_path}")
    torch.save(converted_state, output_path)
    
    logger.info("Conversion completed successfully")
    
    # Print some model information
    logger.info("Model information:")
    # Handle both object and dictionary access
    if hasattr(model_args, "extractor_mode"):
        logger.info(f"- Feature extractor mode: {model_args.extractor_mode}")
    elif isinstance(model_args, dict) and "extractor_mode" in model_args:
        logger.info(f"- Feature extractor mode: {model_args['extractor_mode']}")
        
    if hasattr(model_args, "encoder_layers"):
        logger.info(f"- Encoder layers: {model_args.encoder_layers}")
    elif isinstance(model_args, dict) and "encoder_layers" in model_args:
        logger.info(f"- Encoder layers: {model_args['encoder_layers']}")
        
    if hasattr(model_args, "encoder_embed_dim"):
        logger.info(f"- Encoder embed dim: {model_args.encoder_embed_dim}")
    elif isinstance(model_args, dict) and "encoder_embed_dim" in model_args:
        logger.info(f"- Encoder embed dim: {model_args['encoder_embed_dim']}")
        
    if hasattr(model_args, "encoder_attention_heads"):
        logger.info(f"- Encoder attention heads: {model_args.encoder_attention_heads}")
    elif isinstance(model_args, dict) and "encoder_attention_heads" in model_args:
        logger.info(f"- Encoder attention heads: {model_args['encoder_attention_heads']}")
        
    if hasattr(task_args, "sample_rate"):
        logger.info(f"- Sample rate: {task_args.sample_rate}")
    elif isinstance(task_args, dict) and "sample_rate" in task_args:
        logger.info(f"- Sample rate: {task_args['sample_rate']}")

def main():
    args = parse_args()
    convert_hubert_model(args.fairseq_path, args.output_path)

if __name__ == "__main__":
    main() 