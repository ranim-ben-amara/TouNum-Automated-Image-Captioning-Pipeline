# Tounum - Automated Image Captioning Pipeline

A comprehensive three-stage deep learning pipeline that classifies, denoises, and generates captions for images automatically.

## Overview

This project implements a production-ready pipeline combining computer vision and natural language processing to:

1. **Classify** images as photos or non-photos
2. **Denoise** images for improved quality
3. **Generate** natural language captions describing the image content

## Architecture

### Stage 1: Image Classifier (Deliverable 1)

- **Model**: Custom CNN classifier
- **Purpose**: Determines whether input is a photograph or not
- **Input**: 128×128 RGB images
- **Output**: Binary classification with probability score

### Stage 2: Image Denoiser (Deliverable 2)

- **Model**: Custom denoising neural network
- **Purpose**: Removes noise and artifacts from images for better downstream processing
- **Input**: 256×256 RGB images
- **Output**: Denoised 256×256 images

### Stage 3: Caption Generator

- **Vision Encoder**: InceptionV3 (pre-trained on ImageNet)
  - Extracts visual features from images (299×299)
  - Produces fixed-size feature embeddings
- **Architecture**: Transformer-based encoder-decoder
  - **Encoder**: Processes CNN features with Multi-Head Attention
  - **Decoder**: Generates captions sequentially using:
    - Self-attention over decoded tokens
    - Cross-attention over image features
    - Feed-forward networks for refinement
- **Tokenization**:
  - Vocabulary size: 15,000 words
  - Max caption length: 40 tokens
  - Embedding dimension: 512
  - Transformer units: 512

## Files

- **`finalPipeline_2.ipynb`** - The complete, integrated pipeline notebook with all three stages and visualization

### Required Model Artifacts (in `../All_models/` directory)

- `deliverable1.keras` - Image classifier model
- `deliverable2_denoiser_model.keras` - Denoiser model
- `model.weights.h5` - Caption generator weights
- `text_vocab.json` - Tokenizer vocabulary

## Features

- **Multi-stage processing**: Classification → Denoising → Captioning
- **Transformer architecture**: Modern attention-based caption generation
- **Visual output**: Side-by-side comparison of original and denoised images
- **Probability scoring**: Classification confidence metrics
- **Batch-capable**: Can process individual images or batches
- **Pre-trained encoders**: InceptionV3 backbone for robust feature extraction

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- JSON support
- Python 3.7+

## Usage

```python
# Basic example
result = run_full_pipeline_visual("path/to/image.jpg")
print(result)
# Output: {
#   "is_photo": True,
#   "photo_prob": 0.95,
#   "caption": "A sunset over mountains"
# }
```

### Processing Steps

1. **Load Image** → Read and parse image file
2. **Classify** → Predict if image is a photograph
3. **Denoise** → Remove noise and artifacts
4. **Preprocess** → Resize and normalize for caption model (299×299)
5. **Extract Features** → InceptionV3 CNN feature extraction
6. **Generate Caption** → Transformer sequentially predicts tokens until `[end]` token
7. **Visualize** → Display original, denoised, and caption results

## Pipeline Output

Each processing run returns:

- **is_photo**: Boolean classification result
- **photo_prob**: Float (0.0-1.0) classification confidence
- **caption**: Generated caption string

Visual output shows:

- Left panel: Original image with classification label
- Right panel: Denoised image with generated caption

## Notes

- Image dimensions are automatically resized for each stage
- Captions terminate at max length (40 tokens) or `[end]` token, whichever comes first
- Model files must exist in `../All_models/` relative to notebook directory
- Processing time depends on image size and available GPU resources
