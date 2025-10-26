"""
Key improvements needed in your Colab training:

1. REDUCE LEARNING RATE: 1e-4 → 5e-5
2. INCREASE EPOCHS: 3 → 5
3. REDUCE BATCH SIZE: 4 → 2 (for stability)
4. INCREASE GRADIENT ACCUMULATION: 4 → 8
5. ADD EVALUATION STRATEGY
6. IMPROVE GENERATION PARAMETERS
7. ADD REPETITION PENALTY
"""

# Updated training cell for Colab notebook:

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Data collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# IMPROVED Training arguments
training_args = TrainingArguments(
    output_dir="./sft_results_improved",
    num_train_epochs=5,  # Increased from 3
    per_device_train_batch_size=2,  # Reduced from 4 for stability
    gradient_accumulation_steps=8,  # Increased from 4
    learning_rate=5e-5,  # Reduced from 1e-4
    warmup_steps=200,  # Increased warmup
    weight_decay=0.01,
    fp16=True,
    save_steps=250,  # More frequent saves
    logging_steps=50,  # More frequent logging
    eval_steps=250,  # Add evaluation
    evaluation_strategy="steps",
    load_best_model_at_end=True,  # Load best checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,  # Keep more checkpoints
    dataloader_num_workers=0,  # Reduce for stability
    remove_unused_columns=False,
    report_to="wandb" if use_wandb else "none",
    run_name="flan-t5-sft-improved",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting IMPROVED training...")
trainer.train()

# Save the improved model
trainer.save_model("./sft_trained_model_improved")
tokenizer.save_pretrained("./sft_trained_model_improved")
print("✅ Improved model saved!")

# IMPROVED generation parameters for testing
def generate_improved_response(model, tokenizer, prompt, device='cpu'):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != 'cpu':
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            do_sample=False,  # Use greedy decoding
            repetition_penalty=1.2,  # Prevent repetition
            length_penalty=1.0,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
