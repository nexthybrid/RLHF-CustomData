"""
Improved Training Configuration for Colab
Based on the test results, here are better training parameters
"""

# Better training configuration for Colab
IMPROVED_CONFIG = {
    'num_train_epochs': 5,  # Increase from 3 to 5
    'per_device_train_batch_size': 2,  # Reduce from 4 to 2 for stability
    'gradient_accumulation_steps': 8,  # Increase from 4 to 8
    'learning_rate': 5e-5,  # Reduce from 1e-4 to 5e-5
    'warmup_steps': 200,  # Increase warmup
    'weight_decay': 0.01,
    'fp16': True,
    'save_steps': 250,  # Save more frequently
    'logging_steps': 50,  # Log more frequently
    'eval_steps': 250,  # Add evaluation
    'evaluation_strategy': "steps",
    'load_best_model_at_end': True,
    'metric_for_best_model': "eval_loss",
    'greater_is_better': False,
    'save_total_limit': 3,  # Keep more checkpoints
    'dataloader_num_workers': 0,  # Reduce for stability
    'remove_unused_columns': False,
    'report_to': "wandb" if use_wandb else "none",
    'run_name': "flan-t5-sft-improved",
}

# Better generation parameters
IMPROVED_GENERATION = {
    'max_length': 512,
    'num_beams': 4,
    'early_stopping': True,
    'do_sample': False,  # Use greedy decoding for consistency
    'repetition_penalty': 1.2,  # Increase repetition penalty
    'length_penalty': 1.0,
    'no_repeat_ngram_size': 3,  # Prevent 3-gram repetition
}
