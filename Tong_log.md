# Tong's Log (of What I Did)

## Setup

1. Virtual environment
Set up using pipenv with python 3.11.9

2. Packages
```bash
pip install torch torchvision torchaudio  # Apple Silicon version
pip install transformers datasets accelerate peft trl
pip install rouge-score nltk sacrebleu wandb
pip install sentencepiece
```

## Development

1. Created folders
```bash
mkdir data sft_data
```

2. Downloaded the MetaMathQA dataset

From `https://huggingface.co/datasets/meta-math/MetaMathQA/tree/main` into `RLHF-CustomData/sft_data/`

3. Ran `prep_dataset.py` to prepare a small dataset for training.

4. Download the Flan-T5-base Locally

Model weights: pytorch_model.bin (~500MB) or model.safetensors (safer, same size).
Configuration: config.json.
Tokenizer files: tokenizer.json, tokenizer_config.json, special_tokens_map.json, spiece.model (for SentencePiece).

5. Run the `train_sft.py` script

You may need to login/register a wandb account.