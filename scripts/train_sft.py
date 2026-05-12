#!/usr/bin/env python3
"""Train SFTFlowMidblockModel on reasoning data using HF Trainer.

Usage (smoke test):
    python scripts/train_sft.py --config configs/issue-9/sft_flow_midblock_3060.yaml

Usage (full run):
    python scripts/train_sft.py --config configs/issue-9/sft_flow_midblock.yaml
"""

import argparse
import logging
import os
import sys

import torch
import yaml
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_from_disk

from src.model.sft_flow_midblock import SFTFlowMidblockModel
from src.training.sft_utils import (
    MidblockMetricsCallback,
    validate_model_for_training,
    estimate_training_budget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SFT Flow Matcher")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 training")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 1337)
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found, using CPU (training will be very slow)")

    # 1. Load tokenizer
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded: pad_token={tokenizer.pad_token!r}")

    # 2. Load model with warm-start
    logger.info("Loading SFTFlowMidblockModel...")
    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_path = checkpoint_cfg.get("path", "models/p3_d3_mix_c/checkpoint.pth")

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = SFTFlowMidblockModel(
        model_name=model_cfg["name"],
        start_layer=model_cfg.get("start_layer", 8),
        end_layer=model_cfg.get("end_layer", 11),
        thinking_level=model_cfg.get("thinking_level", 32),
        checkpoint_path=checkpoint_path,
        torch_dtype=dtype,
    )
    logger.info(f"Model created: {model.trainable_params:,} trainable, {model.frozen_params:,} frozen")

    # 3. Validate model setup
    validate_model_for_training(model)

    # 4. Move to device
    model = model.to(device)

    # 5. Load datasets
    data_cfg = config["data"]
    train_dir = os.path.join(data_cfg["processed_dir"], "train")
    eval_dir = os.path.join(data_cfg["processed_dir"], "eval")
    logger.info(f"Loading train dataset from {train_dir}")
    train_dataset = load_from_disk(train_dir)
    logger.info(f"Loading eval dataset from {eval_dir}")
    eval_dataset = load_from_disk(eval_dir)

    # 6. Budget estimation
    budget = estimate_training_budget(
        num_sequences=len(train_dataset),
        seq_length=data_cfg.get("max_seq_length", 8192),
        thinking_level=model_cfg.get("thinking_level", 32),
        batch_size=config["training"].get("per_device_train_batch_size", 1),
        grad_accum=config["training"].get("gradient_accumulation_steps", 1),
        num_epochs=config["training"].get("num_train_epochs", 1),
    )

    # 7. Set up HF Trainer
    training_cfg = config["training"]
    output_dir = training_cfg["output_dir"]
    run_name = training_cfg.get("run_name", "sft_flow_midblock")

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        # Batch
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        # Precision
        fp16=False,
        bf16=not args.fp32 and torch.cuda.is_bf16_supported(),
        fp16_full_eval=False,
        # Schedule
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        max_steps=training_cfg.get("max_steps", -1),
        # Optimizer
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        adam_beta1=training_cfg.get("adam_beta1", 0.9),
        adam_beta2=training_cfg.get("adam_beta2", 0.95),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        # Checkpointing
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        load_best_model_at_end=training_cfg.get("eval_strategy", "steps") != "no",
        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=training_cfg.get("logging_steps", 10),
        report_to=training_cfg.get("report_to", ["tensorboard"]),
        # Evaluation
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 500),
        # Misc
        seed=seed,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        dataloader_drop_last=training_cfg.get("dataloader_drop_last", False),
        # Resume
        resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MidblockMetricsCallback()],
    )

    # 8. Train
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    # 9. Save final model
    logger.info("Saving final model...")
    midblock_save_path = os.path.join(output_dir, "midblock_final.pth")
    torch.save(model.midblock.state_dict(), midblock_save_path)
    logger.info(f"Midblock weights saved to {midblock_save_path}")

    full_save_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), full_save_path)
    logger.info(f"Full model saved to {full_save_path}")

    # 10. Metrics summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total steps:       {train_result.global_step}")
    logger.info(f"  Training loss:     {train_result.training_loss:.4f}")
    logger.info(f"  Total FLOPs est.:  {getattr(train_result, 'total_flos', 0):.2e}")
    logger.info(f"  Output directory:  {output_dir}")


if __name__ == "__main__":
    main()
