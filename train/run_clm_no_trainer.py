#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from contextlib import nullcontext

import datasets
import transformers
from transformers import get_scheduler
import torch
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_metric

# Import MoST model and tokenizer
from configuration_MoST import MoSTConfig
from modeling_most import MoSTForCausalLM
from tokenization_most_fast import MoSTTokenizerFast
from MoST_dataset.multimodal_dataset import create_dataloaders

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.50.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--train_asr_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing ASR training data",
    )
    parser.add_argument(
        "--train_tts_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing TTS training data",
    )
    parser.add_argument(
        "--val_asr_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing ASR validation data",
    )
    parser.add_argument(
        "--val_tts_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing TTS validation data",
    )
    parser.add_argument(
        "--train_text_only_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing text-only training data",
    )
    parser.add_argument(
        "--val_text_only_dirs",
        type=str,
        default=None,
        help="Comma separated list of directories containing text-only validation data",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="Directory to store the cached dataset",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for inputs",
    )
    parser.add_argument(
        "--asr_tts_mix_ratio",
        type=float,
        default=0.5,
        help="Ratio of ASR to TTS samples (0.5 = equal mix)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloaders",
    )
    
    # Training parameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--evaluate_every", type=int, default=100)
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="Whether to use fp16 16-bit (mixed) precision training"
    )
    parser.add_argument(
        "--bf16", 
        action="store_true", 
        help="Whether to use bf16 mixed precision training"
    )
    args = parser.parse_args()

    # Convert string arguments to appropriate types
    if args.train_asr_dirs:
        args.train_asr_dirs = args.train_asr_dirs.split(",")
    if args.train_tts_dirs:
        args.train_tts_dirs = args.train_tts_dirs.split(",")
    if args.val_asr_dirs:
        args.val_asr_dirs = args.val_asr_dirs.split(",")
    if args.val_tts_dirs:
        args.val_tts_dirs = args.val_tts_dirs.split(",")
    if args.train_text_only_dirs:
        args.train_text_only_dirs = args.train_text_only_dirs.split(",")
    if args.val_text_only_dirs:
        args.val_text_only_dirs = args.val_text_only_dirs.split(",")

    # Sanity checks
    if not any([args.train_asr_dirs, args.train_tts_dirs, args.train_text_only_dirs]):
        raise ValueError("Need at least one training data directory (ASR, TTS, or text-only).")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args

GB= 1024**3

def get_memory_stats():
    alloc = torch.cuda.memory_allocated() / GB
    max_alloc = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    max_reserved = torch.cuda.max_memory_reserved() / GB
    return alloc, max_alloc, reserved, max_reserved

def main():
    args = parse_args()

    #################
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    deepspeed_plugin = DeepSpeedPlugin(
        gradient_clipping=1.0,
        zero_stage=2,
        zero3_save_16bit_model=True,
        offload_optimizer_device='cpu',
        offload_param_device='cpu'
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        mixed_precision="bf16" if args.bf16 else "fp16" if args.fp16 else "no",
        **accelerator_log_kwargs
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"{accelerator.state}\n rank: {accelerator.process_index}/{accelerator.num_processes}: "
                f"is_main_process: {accelerator.is_main_process}, is_local_main_process: {accelerator.is_local_main_process}", 
                main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # TODO: Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    #################
    # Prepare model & tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = MoSTTokenizerFast.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    
    # Load model configuration
    config = MoSTConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    
    # Create model
    model = MoSTForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=args.trust_remote_code
    )

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats after initializing model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G", 
        main_process_only=False)
    logger.info(
        f"Model emb size: {model.get_input_embeddings().weight.shape[0]}"
        f", tokenizer vocab size: {len(tokenizer)}")
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Create dataloaders
    train_dataloader, eval_asr_dataloader, eval_tts_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        asr_dirs=args.train_asr_dirs,
        tts_dirs=args.train_tts_dirs,
        train_asr_file_patterns="*train*.json",
        train_tts_file_patterns="*train*.json",
        val_asr_file_patterns=["*dev*.json", "*valid*.json"],
        val_tts_file_patterns=["*dev*.json", "*valid*.json"],
        max_seq_length=args.max_seq_length,
        asr_tts_mix_ratio=args.asr_tts_mix_ratio,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed
    )
    
    # Check if at least one validation dataloader exists
    has_eval = eval_asr_dataloader is not None or eval_tts_dataloader is not None

    if train_dataloader is None:
        raise ValueError("No training data provided. Please specify at least one of --train_asr_dirs or --train_tts_dirs")

    #################
    # Prepare optimizer and scheduler
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    
    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        # Calculate total steps based on dataset size and epochs
        args.max_train_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * args.num_train_epochs
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Prepare everything with `accelerator`.
    # Ensure eval dataloaders are only prepared if they exist
    prepared_components = [model, optimizer, train_dataloader]
    if eval_asr_dataloader:
        prepared_components.append(eval_asr_dataloader)
    if eval_tts_dataloader:
        prepared_components.append(eval_tts_dataloader)
    prepared_components.append(lr_scheduler)
    
    prepared_result = accelerator.prepare(*prepared_components)

    model = prepared_result[0]
    optimizer = prepared_result[1]
    train_dataloader = prepared_result[2]
    eval_asr_dataloader = prepared_result[3] if eval_asr_dataloader else None
    eval_tts_dataloader = prepared_result[4 if eval_asr_dataloader else 3] if eval_tts_dataloader else None
    lr_scheduler = prepared_result[-1]

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model}")
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats before training: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
        , main_process_only=False)
    for idx, batch in enumerate(train_dataloader):
        logger.info(f"rank {accelerator.process_index} batch {idx}: {batch['input_ids'][0, :5].tolist()}", main_process_only=False)
        if idx == 2:
            break

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("most-training", experiment_config)

    # Load WER metric
    wer_metric = None
    if eval_asr_dataloader:
        try:
            wer_metric = load_metric("wer")
            logger.info("WER metric loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load WER metric: {e}. WER will not be computed.")

    completed_steps = 0
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        dirs = [f.path for f in os.scandir(args.resume_from_checkpoint) if f.is_dir() and f.name.startswith("step_")]
        if len(dirs) > 0:
            dirs.sort(key=os.path.getctime)
            checkpoint_path = dirs[-1]
            path = os.path.basename(checkpoint_path)

            accelerator.load_state(checkpoint_path)
            # Extract `step_{i}`
            training_difference = os.path.splitext(path)[0]

            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            completed_steps = resume_step // args.gradient_accumulation_steps
            logger.info(
                f"Resumed from checkpoint: {checkpoint_path}, resume steps (w. grad acc): {resume_step}, "
                f"completed_steps: {completed_steps}"
            )
        else:
            logger.warning(
                f"Please be aware that resume_from_checkpoint is specified as {args.resume_from_checkpoint}, "
                f"but no ckpt is detected"
            )

    if not args.skip_train:
        #################
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        model.train()
        if args.with_tracking:
            step_loss = torch.zeros(1, device=model.device, dtype=torch.float)
        if args.resume_from_checkpoint and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # Filter batch to only include expected model inputs
                model_inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                outputs = model(**model_inputs)
                loss = outputs['loss']

                if args.with_tracking:
                    step_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking:
                    step_loss /= args.gradient_accumulation_steps
                    global_loss = accelerator.reduce(step_loss, reduction='mean')
                    log_info = {"train_loss": global_loss.item(),}
                    for lr_idx, lr in enumerate(lr_scheduler.get_last_lr()):
                        log_info[f"lr_{lr_idx}"] = lr
                    
                    accelerator.log(log_info, step=completed_steps)
                    step_loss.zero_()

                if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                if has_eval and completed_steps % args.evaluate_every == 0:
                    eval_metrics = {}
                    
                    # ASR Evaluation
                    if eval_asr_dataloader:
                        asr_losses = []
                        predictions = []
                        references = []
                        model.eval()
                        for step, batch in tqdm(
                            enumerate(eval_asr_dataloader), desc=f"ASR Evaluation at step {completed_steps}",
                            disable=not accelerator.is_local_main_process, total=len(eval_asr_dataloader)
                        ):
                            with torch.no_grad():
                                model_inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                                outputs = model(**model_inputs)
                                loss = outputs['loss']
                                logits = outputs['logits']
                            
                            gathered_loss = accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                            asr_losses.append(gathered_loss)

                            # Generate predictions for WER
                            if wer_metric:
                                labels = model_inputs['labels']
                                # Get predicted token ids
                                pred_ids = torch.argmax(logits, dim=-1)
                                
                                # Gather across devices
                                gathered_preds = accelerator.gather_for_metrics(pred_ids)
                                gathered_labels = accelerator.gather_for_metrics(labels)

                                # Decode predictions and labels, skipping special tokens and padding
                                # Handle potential padding in labels (-100)
                                gathered_labels[gathered_labels == -100] = tokenizer.pad_token_id
                                
                                decoded_preds = tokenizer.batch_decode(gathered_preds, skip_special_tokens=True)
                                decoded_labels = tokenizer.batch_decode(gathered_labels, skip_special_tokens=True)
                                
                                # Add to WER metric (ensure main process handles this after gather)
                                if accelerator.is_main_process:
                                    wer_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                                    # Store for potential inspection if needed, might consume memory
                                    # predictions.extend(decoded_preds)
                                    # references.extend(decoded_labels)

                        model.train()

                        # Compute evaluation metrics
                        asr_losses = torch.cat(asr_losses)
                        try:
                            eval_asr_loss = torch.mean(asr_losses)
                            asr_perplexity = math.exp(eval_asr_loss)
                        except OverflowError:
                            asr_perplexity = float('inf')
                            eval_asr_loss = torch.tensor(float('inf'))

                        logger.info(f"step: {completed_steps} ASR perplexity: {asr_perplexity} eval_asr_loss: {eval_asr_loss}")
                        eval_metrics.update({"asr_perplexity": asr_perplexity, "eval_asr_loss": eval_asr_loss.item()})

                        # Compute WER
                        if wer_metric and accelerator.is_main_process:
                            wer_score = wer_metric.compute()
                            logger.info(f"step: {completed_steps} ASR WER: {wer_score}")
                            eval_metrics.update({"asr_wer": wer_score})
                            # Reset metric for next evaluation cycle if needed by the metric implementation
                            # wer_metric = load_metric("wer") # Re-load or reset if necessary
                    
                    # TTS Evaluation
                    if eval_tts_dataloader:
                        tts_losses = []
                        model.eval()
                        for step, batch in tqdm(
                            enumerate(eval_tts_dataloader), desc=f"TTS Evaluation at step {completed_steps}",
                            disable=not accelerator.is_local_main_process, total=len(eval_tts_dataloader)
                        ):
                            with torch.no_grad():
                                model_inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                                outputs = model(**model_inputs)
                            loss = outputs['loss'].repeat(args.per_device_eval_batch_size)
                            tts_losses.append(accelerator.gather(loss))
                        model.train()

                        tts_losses = torch.cat(tts_losses)
                        try:
                            eval_tts_loss = torch.mean(tts_losses)
                            tts_perplexity = math.exp(eval_tts_loss)
                        except OverflowError:
                            tts_perplexity = float('inf')
                            eval_tts_loss = torch.tensor(float('inf')) # Handle potential overflow in loss as well
                            
                        logger.info(f"step: {completed_steps} TTS perplexity: {tts_perplexity} eval_tts_loss: {eval_tts_loss}")
                        eval_metrics.update({"tts_perplexity": tts_perplexity, "eval_tts_loss": eval_tts_loss.item()})

                    if args.with_tracking and eval_metrics:
                        accelerator.log(eval_metrics, step=completed_steps)
                            
            if completed_steps >= args.max_train_steps:
                break
    
    # Final Evaluation
    if has_eval:
        logger.info("***** Running Final Evaluation *****")
        final_eval_metrics = {}

        # Final ASR Evaluation
        if eval_asr_dataloader:
            asr_losses = []
            predictions = []
            references = []
            # Reset WER metric for final evaluation if loaded
            if wer_metric and accelerator.is_main_process:
                 try:
                     wer_metric = load_metric("wer") # Re-load to ensure clean state
                 except Exception as e:
                     logger.warning(f"Could not reload WER metric for final eval: {e}")
                     wer_metric = None # Disable WER if reload fails

            model.eval()
            for step, batch in tqdm(
                enumerate(eval_asr_dataloader), desc="Final ASR Evaluation",
                disable=not accelerator.is_local_main_process, total=len(eval_asr_dataloader)
            ):
                with torch.no_grad():
                    model_inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                    outputs = model(**model_inputs)
                    loss = outputs['loss']
                    logits = outputs['logits']
                
                gathered_loss = accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                asr_losses.append(gathered_loss)
                
                # Generate predictions for WER
                if wer_metric:
                    labels = model_inputs['labels']
                    pred_ids = torch.argmax(logits, dim=-1)
                    
                    gathered_preds = accelerator.gather_for_metrics(pred_ids)
                    gathered_labels = accelerator.gather_for_metrics(labels)

                    gathered_labels[gathered_labels == -100] = tokenizer.pad_token_id
                    
                    decoded_preds = tokenizer.batch_decode(gathered_preds, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(gathered_labels, skip_special_tokens=True)
                    
                    if accelerator.is_main_process:
                        wer_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                        # predictions.extend(decoded_preds)
                        # references.extend(decoded_labels)
            
            # Compute final ASR metrics
            asr_losses = torch.cat(asr_losses)
            try:
                final_eval_asr_loss = torch.mean(asr_losses)
                final_asr_perplexity = math.exp(final_eval_asr_loss)
            except OverflowError:
                final_asr_perplexity = float('inf')
                final_eval_asr_loss = torch.tensor(float('inf'))

            logger.info(f"Final ASR perplexity: {final_asr_perplexity} final_eval_asr_loss: {final_eval_asr_loss}")
            final_eval_metrics.update({"final_asr_perplexity": final_asr_perplexity, "final_eval_asr_loss": final_eval_asr_loss.item()})

            # Compute final WER
            if wer_metric and accelerator.is_main_process:
                final_wer_score = wer_metric.compute()
                logger.info(f"Final ASR WER: {final_wer_score}")
                final_eval_metrics.update({"final_asr_wer": final_wer_score})

        # Final TTS Evaluation
        if eval_tts_dataloader:
            tts_losses = []
            model.eval()
            for step, batch in tqdm(
                enumerate(eval_tts_dataloader), desc="Final TTS Evaluation",
                disable=not accelerator.is_local_main_process, total=len(eval_tts_dataloader)
            ):
                with torch.no_grad():
                    model_inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                    outputs = model(**model_inputs)
                loss = outputs['loss'].repeat(args.per_device_eval_batch_size)
                tts_losses.append(accelerator.gather(loss))

            tts_losses = torch.cat(tts_losses)
            try:
                final_eval_tts_loss = torch.mean(tts_losses)
                final_tts_perplexity = math.exp(final_eval_tts_loss)
            except OverflowError:
                final_tts_perplexity = float('inf')
                final_eval_tts_loss = torch.tensor(float('inf'))
            
            logger.info(f"Final TTS perplexity: {final_tts_perplexity} final_eval_tts_loss: {final_eval_tts_loss}")
            final_eval_metrics.update({"final_tts_perplexity": final_tts_perplexity, "final_eval_tts_loss": final_eval_tts_loss.item()})

        if args.with_tracking and final_eval_metrics:
            accelerator.log(final_eval_metrics, step=completed_steps) # Log final metrics at the last step

    alloc, max_allc, resv, max_resv = get_memory_stats()
    logger.info(
        f"Memory stats on exiting: Alloc: {alloc:.2f} G / {max_allc:.2f} G, Resrv: {resv:.2f} G / {max_resv:.2f} G"
        , main_process_only=False)

    if args.output_dir is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo_id = args.hub_model_id if args.hub_model_id else os.path.basename(args.output_dir)
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
        logger.info(f"Saving model to {args.output_dir} done!", main_process_only=False)
        accelerator.wait_for_everyone()
    
    if args.with_tracking:        
        accelerator.end_training()


if __name__ == "__main__":
    main()