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
pip install matplotlib
```

## Development

1. Created folders
```bash
mkdir data sft_data
```

2. Downloaded the MetaMathQA dataset

From `https://huggingface.co/datasets/meta-math/MetaMathQA/tree/main` into `RLHF-CustomData/sft_model/`

3. Ran `prep_dataset.py` to prepare a small dataset for training.

4. Download the Flan-T5-base Locally

Model weights: pytorch_model.bin (~500MB) or model.safetensors (safer, same size).
Configuration: config.json.
Tokenizer files: tokenizer.json, tokenizer_config.json, special_tokens_map.json, spiece.model (for SentencePiece).

5. Run the `train_sft.py` script

Initially taking 100 entries only; now taking 500 entries. Initially training for 1 epoch only; now training for 3 epochs.

You may need to login/register a wandb account.

6. Runs the `test_models.py` script

This shows comparison between the original and fine-tuned model output, and the ROUGE-L scores.

## Updates

10/25 - Developed Colab version (train on Colab, download the model, inference locally)
