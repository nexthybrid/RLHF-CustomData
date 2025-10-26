# Colab Model Local Inference Guide

This guide explains how to use the downloaded Colab-trained model for local inference.

## 📁 File Structure

After downloading the model from Colab, you should have:
```
your_project/
├── sft_trained_model/          # Extracted from downloaded zip
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── test_models_colab.py        # Local inference script
├── sft_data/
│   └── train.json             # Your training data
└── ...
```

## 🚀 Quick Start

### 1. Extract Downloaded Model
```bash
# Extract the downloaded zip file
unzip sft_trained_model.zip
```

### 2. Install Dependencies
```bash
pip install transformers torch rouge-score
```

### 3. Run Basic Inference
```bash
python test_models_colab.py --model_path ./sft_trained_model
```

### 4. Run with Comparison
```bash
python test_models_colab.py --model_path ./sft_trained_model --compare
```

### 5. Run with GPU (if available)
```bash
python test_models_colab.py --model_path ./sft_trained_model --device cuda --compare
```

## 🔧 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to Colab-trained model directory | `./sft_trained_model` |
| `--test_data` | Path to test data JSON file | `./sft_data/train.json` |
| `--device` | Device to run inference on (`cpu` or `cuda`) | `cpu` |
| `--compare` | Compare with original Flan-T5 model | `False` |

## 📊 Example Output

```
🚀 Starting Colab model evaluation
Device: cpu
Model path: ./sft_trained_model
Loading Colab-trained model from: ./sft_trained_model
✅ Model loaded successfully
Model parameters: 247,577,856

🧪 Evaluating model performance...

--- Test 1 ---
Prompt: solve: What is 2 + 2?
Response: 4
ROUGE-L Score: 1.0000
Ground Truth: The answer is 4...

--- Test 2 ---
Prompt: solve: If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?
Response: 180 miles
ROUGE-L Score: 0.8571
Ground Truth: The train will travel 180 miles in 3 hours...

🔄 Comparing Colab-trained vs Original model...

--- Comparison 1 ---
Prompt: solve: What is 2 + 2?
Colab-trained: 4
Original: two
Colab ROUGE-L: 1.0000
Original ROUGE-L: 0.0000
Improvement: +1.0000
```

## 🎯 Features

### **Model Evaluation**
- **Multiple test prompts** for comprehensive testing
- **ROUGE-L scoring** against ground truth
- **Performance metrics** and statistics

### **Model Comparison**
- **Side-by-side comparison** with original Flan-T5
- **Improvement metrics** showing training effectiveness
- **Quality assessment** of fine-tuning results

### **Flexible Usage**
- **CPU/GPU support** for different hardware
- **Custom test data** support
- **Command-line interface** for easy automation

## 🔍 Troubleshooting

### **Model Not Found**
```bash
# Check if model directory exists
ls -la ./sft_trained_model/

# Verify model files
ls -la ./sft_trained_model/*.json
ls -la ./sft_trained_model/*.safetensors
```

### **CUDA Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if CUDA not available
python test_models_colab.py --device cpu
```

### **Missing Dependencies**
```bash
# Install all required packages
pip install transformers torch rouge-score datasets accelerate
```

## 📈 Performance Tips

1. **Use GPU** for faster inference: `--device cuda`
2. **Batch processing** for multiple prompts
3. **Model quantization** for memory efficiency
4. **Caching** for repeated evaluations

## 🔄 Integration with Existing Code

You can also integrate the Colab model into your existing `test_models.py`:

```python
# Load Colab model instead of local SFT model
colab_tokenizer = T5Tokenizer.from_pretrained('./sft_trained_model')
colab_model = T5ForConditionalGeneration.from_pretrained('./sft_trained_model')

# Use in your existing pipeline
colab_pipe = pipeline('text2text-generation', model=colab_model, tokenizer=colab_tokenizer)
output = colab_pipe("solve: What is 2 + 2?")[0]['generated_text']
```

## 📞 Support

If you encounter issues:
1. **Check file paths** and permissions
2. **Verify model files** are complete
3. **Update dependencies** to latest versions
4. **Use CPU mode** if GPU issues persist

---

**Happy Testing! 🚀**
