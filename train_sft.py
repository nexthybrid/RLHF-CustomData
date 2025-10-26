from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load prepped data
dataset = load_dataset('json', data_files='sft_data/train.json')

# Load tokenizer and model from local directory
tokenizer = T5Tokenizer.from_pretrained('./flan-t5-base', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('./flan-t5-base')

def tokenize(examples):
    inputs = tokenizer(examples['prompt'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(examples['response'], max_length=256, truncation=True, padding='max_length', return_tensors='pt').input_ids
    inputs['labels'] = labels
    return inputs

tokenized_ds = dataset.map(tokenize, batched=True)

# Subset for local trial
tokenized_ds = tokenized_ds['train'].shuffle().select(range(500))  # Increase to 500 samples

training_args = TrainingArguments(
    output_dir='sft_output',
    num_train_epochs=3,  # Increase to 3 epochs
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
    fp16=False,
    use_mps_device=torch.backends.mps.is_available(),
    report_to="wandb"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds)
trainer.train()

model.save_pretrained('sft_model')
tokenizer.save_pretrained('sft_model')
print("SFT trial done! Model saved.")