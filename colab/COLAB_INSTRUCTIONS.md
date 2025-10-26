# Google Colab Setup Instructions for SFT Training

This guide will help you run your SFT (Supervised Fine-Tuning) training on Google Colab using their powerful GPU resources.

## 📋 Prerequisites

1. **Google Account** - You need a Google account to access Colab
2. **Training Data** - Your `train.json` file from the `sft_data/` directory
3. **WandB Account** (Optional) - For training monitoring

## 🚀 Step-by-Step Instructions

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click "New Notebook" or "File" → "New notebook"

### Step 2: Upload the Notebook
1. Download the `SFT_Training_Colab.ipynb` file from your project
2. In Colab, go to "File" → "Upload notebook"
3. Select and upload the `SFT_Training_Colab.ipynb` file

### Step 3: Enable GPU Runtime
1. In Colab, go to "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to **GPU** (T4 or V100)
3. Click "Save"

### Step 4: Run the Notebook
Execute the cells in order by clicking the play button (▶️) or using `Shift + Enter`:

#### Cell 1: Install Packages
- Installs all required dependencies
- Verifies GPU availability
- **Expected output**: CUDA available: True, GPU name and memory info

#### Cell 2: Upload Data
- Click "Choose Files" and select your `train.json` file
- The file will be uploaded and moved to the correct location
- **Expected output**: ✅ Data file uploaded successfully!

#### Cell 3: Setup WandB (Optional)
- If you have a WandB account, run this cell and enter your API key
- If not, you can skip this - training will continue without logging
- **Expected output**: ✅ WandB login successful! (or warning message)

#### Cell 4: Load Model
- Downloads the Flan-T5 base model from Hugging Face
- Moves model to GPU
- **Expected output**: Model loaded, parameter count, device info

#### Cell 5: Load Data
- Loads your training dataset
- Shows sample data structure
- **Expected output**: Dataset loaded with sample count

#### Cell 6: Tokenize Data
- Tokenizes the dataset for training
- Creates training subset (up to 1000 samples)
- **Expected output**: Training dataset prepared with sample count

#### Cell 7: Training Configuration
- Sets up training parameters optimized for Colab
- **Expected output**: Training configuration details

#### Cell 8: Initialize Trainer
- Creates the Hugging Face Trainer
- **Expected output**: Trainer initialized with step count

#### Cell 9: Start Training
- **This is the main training cell - it will take 30-60 minutes**
- Monitor progress in the output
- **Expected output**: Training progress with loss values

#### Cell 10: Save Model
- Saves the trained model and tokenizer
- **Expected output**: Model saved successfully with file list

#### Cell 11: Download Model
- Creates a zip file of your trained model
- Automatically downloads it to your local machine
- **Expected output**: Download started message

#### Cell 12: Test Model (Optional)
- Tests the trained model with a sample prompt
- **Expected output**: Model response to test question

## ⚙️ Configuration Options

### Adjust Training Parameters
You can modify these parameters in Cell 7:

```python
# Training arguments optimized for Colab GPU
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,                    # Number of training epochs
    per_device_train_batch_size=4,        # Batch size (increase if you have more GPU memory)
    warmup_steps=100,                      # Warmup steps
    weight_decay=0.01,                     # Weight decay
    fp16=True,                            # Mixed precision training
    # ... other parameters
)
```

### Adjust Dataset Size
In Cell 6, you can change the training subset size:

```python
train_size = min(1000, len(tokenized_dataset['train']))  # Change 1000 to your desired size
```

## 🔧 Troubleshooting

### Common Issues:

1. **GPU Not Available**
   - Make sure you selected GPU runtime in Step 3
   - Try refreshing the page and re-selecting GPU

2. **Out of Memory Error**
   - Reduce `per_device_train_batch_size` from 4 to 2 or 1
   - Reduce `train_size` in the tokenization cell

3. **Training Too Slow**
   - Ensure `fp16=True` is set (mixed precision)
   - Check that GPU is being used (should show CUDA device)

4. **Data Upload Issues**
   - Make sure your `train.json` file is properly formatted
   - Check that the file contains the expected "prompt" and "response" fields

5. **WandB Login Issues**
   - Skip WandB setup if you don't have an account
   - Training will work without it

### Performance Tips:

1. **Use T4 GPU** (free tier) - Good for small to medium datasets
2. **Use V100 GPU** (Colab Pro) - Better for larger datasets and faster training
3. **Monitor GPU usage** - Check if GPU is being utilized efficiently
4. **Batch size optimization** - Start with batch size 4, adjust based on memory usage

## 📊 Expected Results

- **Training Time**: 30-60 minutes for 1000 samples
- **Model Size**: ~1.5GB (Flan-T5-base)
- **Final Output**: Trained model files in a zip archive

## 🔄 Next Steps

After training completes:

1. **Download the model** using Cell 11
2. **Test locally** by extracting the zip file
3. **Use for inference** in your local environment
4. **Fine-tune further** if needed with different parameters

## 💡 Tips for Better Results

1. **Data Quality**: Ensure your training data is high-quality and diverse
2. **Hyperparameters**: Experiment with different learning rates and batch sizes
3. **Monitoring**: Use WandB to track training progress and compare runs
4. **Validation**: Test your model on unseen data after training

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Ensure all dependencies are properly installed
4. Check Colab's GPU availability and runtime settings

---

**Happy Training! 🚀**
