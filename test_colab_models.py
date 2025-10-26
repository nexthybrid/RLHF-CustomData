"""
Better Test Script for Colab-Trained Model
Uses prompts from actual training data for proper evaluation
"""

import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer

def load_model(model_path):
    """Load the Colab-trained model"""
    print(f"Loading model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    print(f"✅ Model loaded successfully")
    return model, tokenizer

def load_training_data(data_path):
    """Load training data to get actual prompts"""
    print(f"Loading training data from: {data_path}")
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Create ground truth mapping
    ground_truths = {d['prompt']: d['response'] for d in data}
    print(f"✅ Loaded {len(data)} training samples")
    return data, ground_truths

def test_with_training_prompts(model, tokenizer, data, ground_truths, device='cpu'):
    """Test model with actual training prompts"""
    print("\n🧪 Testing with training data prompts...")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Test with first 5 samples from training data
    test_samples = data[:5]
    
    total_rouge = 0
    valid_tests = 0
    
    for i, sample in enumerate(test_samples):
        prompt = sample['prompt']
        ground_truth = sample['response']
        
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Ground Truth: {ground_truth[:100]}...")
        
        # Generate response
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
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model Response: {response[:200]}...")
        
        # Calculate ROUGE score
        rouge_score = scorer.score(ground_truth, response)['rougeL'].fmeasure
        print(f"ROUGE-L Score: {rouge_score:.4f}")
        
        total_rouge += rouge_score
        valid_tests += 1
    
    avg_rouge = total_rouge / valid_tests if valid_tests > 0 else 0
    print(f"\n📊 Average ROUGE-L Score: {avg_rouge:.4f}")
    return avg_rouge

def test_simple_math(model, tokenizer, device='cpu'):
    """Test with simple mathematical problems"""
    print("\n🔢 Testing simple mathematical problems...")
    
    simple_tests = [
        "solve: What is 2 + 2?",
        "solve: What is 5 * 3?",
        "solve: What is 10 - 4?",
        "solve: What is 8 / 2?",
        "solve: What is 3^2?"
    ]
    
    for i, prompt in enumerate(simple_tests):
        print(f"\n--- Simple Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=False,  # Use greedy decoding for simple problems
                temperature=0.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

def main():
    model_path = './sft_trained_model'
    data_path = './sft_data/train.json'
    device = 'cpu'
    
    print("🚀 Starting comprehensive model evaluation")
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load training data
    data, ground_truths = load_training_data(data_path)
    
    # Test with training prompts
    avg_rouge = test_with_training_prompts(model, tokenizer, data, ground_truths, device)
    
    # Test with simple math
    test_simple_math(model, tokenizer, device)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Average ROUGE-L on training prompts: {avg_rouge:.4f}")
    
    if avg_rouge < 0.3:
        print("⚠️  Low ROUGE score suggests model needs more training")
    elif avg_rouge < 0.6:
        print("📈 Moderate ROUGE score - model shows some learning")
    else:
        print("✅ Good ROUGE score - model appears well-trained")
    
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    main()
