from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from rouge_score import rouge_scorer
import json

# Load fine-tuned model and tokenizer
sft_tokenizer = T5Tokenizer.from_pretrained('./sft_model')
sft_model = T5ForConditionalGeneration.from_pretrained('./sft_model')

# Load original model and tokenizer
orig_tokenizer = T5Tokenizer.from_pretrained('./flan-t5-base')
orig_model = T5ForConditionalGeneration.from_pretrained('./flan-t5-base')

# Load ground truths from train.json (map prompts to responses)
with open('sft_data/train.json', 'r') as f:
    data = [json.loads(line) for line in f]
ground_truths = {d['prompt']: d['response'] for d in data}

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Test prompts
test_prompts = [
    "solve: What is the derivative of x^2?",
    "solve: Solve 2x + 3 = 7 for x.",
    "solve: What is the integral of x dx?"
]

# Generate responses and compute ROUGE scores
for prompt in test_prompts:
    # Fine-tuned
    sft_pipe = pipeline('text2text-generation', model=sft_model, tokenizer=sft_tokenizer)
    sft_output = sft_pipe(prompt)[0]['generated_text']

    # Original
    orig_pipe = pipeline('text2text-generation', model=orig_model, tokenizer=orig_tokenizer)
    orig_output = orig_pipe(prompt)[0]['generated_text']

    # Get ground truth (fallback to first if not found)
    gt = ground_truths.get(prompt, next(iter(ground_truths.values())))

    # Compute ROUGE-L scores
    sft_rouge = scorer.score(gt, sft_output)['rougeL'].fmeasure
    orig_rouge = scorer.score(gt, orig_output)['rougeL'].fmeasure

    print(f"Prompt: {prompt}")
    print(f"Ground Truth: {gt}")
    print(f"Fine-tuned: {sft_output}")
    print(f"Original: {orig_output}")
    print(f"ROUGE-L (Fine-tuned): {sft_rouge:.4f}")
    print(f"ROUGE-L (Original): {orig_rouge:.4f}")
    print("---")