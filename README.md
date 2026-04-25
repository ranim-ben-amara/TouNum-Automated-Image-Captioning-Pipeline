# TouNum - Automated Image Captioning Pipeline

TouNum is a notebook-based image understanding project with three deliverables:

1. Binary image classification
2. Image denoising
3. Image caption generation

The repository also includes a combined notebook that chains the three stages into a single inference pipeline.

## Repository Layout

- `Deliverable1_Binary_Image_Classification.ipynb` - trains the photo vs. non-photo classifier
- `Deliverable2_Image_Denoising.ipynb` - trains the denoising model
- `Deliverable3_Image_Captioning.ipynb` - builds and trains the captioning model
- `finalPipeline_2.ipynb` - runs the end-to-end pipeline using the saved artifacts

## What The Pipeline Does

The integrated workflow takes an input image and:

1. Classifies it as a photo or non-photo image
2. Denoises it to improve downstream quality
3. Generates a natural-language caption for the image content

The final notebook also visualizes the original and denoised image outputs.

## Model Artifacts

The notebooks expect trained artifacts to be available outside the repository. The combined pipeline loads these files:

- `deliverable1.keras` - classifier model
- `deliverable2_denoiser_model.keras` - denoiser model
- `model.weights.h5` - captioning model weights
- `text_vocab.json` - tokenizer vocabulary

If your artifact directory is different, update the path configured in `finalPipeline_2.ipynb` before running the notebook.

## Requirements

The notebooks were built for a Python notebook environment with:

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- pandas
- scikit-learn
- scikit-image
- Pillow
- imageio
- requests
- tqdm

## Running The Project

1. Open the deliverable notebook you want to train or inspect.
2. Run the cells in order to build the corresponding model artifact.
3. Place the saved artifacts where `finalPipeline_2.ipynb` expects them.
4. Open `finalPipeline_2.ipynb` and run the pipeline on an input image.

Example usage inside the final notebook:

```python
result = run_full_pipeline_visual("path/to/image.jpg")
print(result)
```

The returned result includes:

- `is_photo` - whether the image was classified as a photo
- `photo_prob` - classifier confidence score
- `caption` - generated caption text

## Notes

- Image sizes are resized automatically per stage.
- Caption generation stops at the end token or the configured maximum length.
- The repository does not include the trained model artifacts or source datasets.
- Notebook execution time depends on the available CPU/GPU environment.
