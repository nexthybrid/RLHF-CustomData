from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# Load fine-tuned model and tokenizer
sft_tokenizer = T5Tokenizer.from_pretrained('./sft_model')
sft_model = T5ForConditionalGeneration.from_pretrained('./sft_model')

# Load original model and tokenizer (using local flan-t5-base files)
orig_tokenizer = T5Tokenizer.from_pretrained('./flan-t5-base')
orig_model = T5ForConditionalGeneration.from_pretrained('./flan-t5-base')

# Test prompts (sample from MetaMathQA)
test_prompts = [
    "solve: What is the derivative of x^2?",
    "solve: Solve 2x + 3 = 7 for x.",
    "solve: What is the integral of x dx?"
]

# Generate responses
for prompt in test_prompts:
    # Fine-tuned
    sft_pipe = pipeline('text2text-generation', model=sft_model, tokenizer=sft_tokenizer)
    sft_output = sft_pipe(prompt)[0]['generated_text']

    # Original
    orig_pipe = pipeline('text2text-generation', model=orig_model, tokenizer=orig_tokenizer)
    orig_output = orig_pipe(prompt)[0]['generated_text']

    print(f"Prompt: {prompt}")
    print(f"Fine-tuned: {sft_output}")
    print(f"Original: {orig_output}")
    print("---")