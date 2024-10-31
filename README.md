# CharGPT

This repository contains code to build a Generative Pretrained Transformer (GPT) model from scratch, inspired by Andrej Karpathy's video tutorial on transformers and GPT architectures. Following concepts from the foundational paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) and OpenAI's GPT-2/GPT-3, this project explores language modeling and the inner workings of transformers, with each component implemented from the ground up.

![GPT Model Architecture](images/gpt_architecture.png)
![Model Outputs](images/outputs.png)

## Project Overview

In this project, I develop a baseline GPT model using PyTorch, following Andrej Karpathy's tutorial structure. The project includes:
- **Data preprocessing** and **tokenization**
- **Bigram language model** as the simplest baseline
- **Self-attention mechanism**, the key to transformer models
- **Multi-headed self-attention** and feedforward layers
- **Transformer blocks** with residual connections and layer normalization
- **Model scaling** and optimization with dropout

Each step is implemented progressively to give a detailed understanding of transformer architecture.

## Acknowledgments

This project is based on [Andrej Karpathy's video tutorial]([https://www.youtube.com/watch?v=...](https://youtu.be/kCc8FmEb1nY?si=EdnmoRoxrOEf8fLo)) on building a GPT model from scratch. His walkthrough provides foundational knowledge and guided this project's development. I recommend watching his earlier "makemore" series to get a solid grounding in autoregressive language modeling and basic PyTorch.

## Usage

1. Make sure you have the required packages installed (I prefer using Conda) (mainly PyTorch)
2. Use your own input.txt or use the provided data (Tiny-Shakespeare Dataset)
3. Run the gpt.py file to train the model and generate the text

## Repository Structure

```plaintext
.
├── README.md                  # Project description and setup instructions
├── input.txt                  # Sample text data used for training and testing
├── images/                    # Images for model architecture and sample outputs
│   └── gpt_architecture.png   # Example: architecture image for model visualization
│   └── outputs.png            # Example: architecture image for model visualization
├── notebooks/                 # Jupyter notebooks for exploration and prototyping
│   └── Char_GPT.ipynb         # Prototyping of the CharGPT model and experimentation
├── char_gpt_model.pth         # Saved weights of the trained CharGPT model
├── generated_text.txt         # Output generated from the model
├── gpt_combined_script.py     # Complete scripts handling model architecture, data loading, training and prediction
├── bigramLM.py                # Baseline model (Bigram language model implementation)
├── model.py                   # Model definition (CharGPT model architecture)
├── train.py                   # Training script for the CharGPT model
├── predict.py                 # Script for generating text with the trained model
├── utils.py                   # Helper functions used across multiple scripts
