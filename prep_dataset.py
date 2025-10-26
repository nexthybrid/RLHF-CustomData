from datasets import load_dataset
import os

# Load local JSON (first 1000 rows for trial)
ds = load_dataset('json', data_files='sft_data/MetaMathQA-395K.json', split='train[:1000]')

def preprocess(examples):
    prompts = [f"solve: {q}" for q in examples['query']]
    responses = examples['response']
    return {'prompt': prompts, 'response': responses}

processed_ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
os.makedirs('sft_data', exist_ok=True)
processed_ds.to_json('sft_data/train.json', orient='records')
print("Dataset prepped! Sample:", processed_ds[0])