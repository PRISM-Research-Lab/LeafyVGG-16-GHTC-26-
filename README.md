# LeafyVGG-16: Transfer Learning for Plant Disease Detection with Cyber Risk Analysis

This repository contains the TensorFlow implementation of **LeafyVGG-16**
for multiclass plant disease classification using an ImageNet-pretrained
VGG16 model. It includes model training, fine-tuning, test-set evaluation,
and adversarial robustness analysis using **Fast Gradient Sign Method
(FGSM)** and **Projected Gradient Descent (PGD)**.

The associated paper has been **accepted at the 2026 IEEE Global
Humanitarian Technology Conference (GHTC)**. Its preprint is available
on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.13.724946v1).

## Model Architecture

LeafyVGG-16 uses the VGG16 convolutional backbone with ImageNet-pretrained
weights and a custom classification head:

- Input: 224 × 224 RGB image.
- VGG16 backbone with its original classification head removed.
- Global Average Pooling.
- Dense layer: 512 units with ReLU activation.
- Batch Normalization.
- Dropout: 0.4.
- Dense layer: 256 units with ReLU activation.
- Dropout: 0.3.
- Output layer: one unit per class with softmax activation.

Training follows two stages:

1. **Classification-head training:** The VGG16 backbone is frozen for
   up to 50 epochs, using Adam with a learning rate of 0.001.
2. **Fine-tuning:** The backbone is unfrozen while its first 15 layers
   remain frozen. Training continues for up to 200 additional epochs,
   using Adam with a learning rate of 0.00001.

The model uses categorical cross-entropy loss. Training includes data
augmentation, model checkpointing, learning-rate reduction, and early
stopping.

## Code Files

| File | Purpose |
|---|---|
| `LeafyVGG-16_PretrainedV1.py` | Trains and fine-tunes LeafyVGG-16, saves models and training history, displays training curves, and evaluates the test set. |
| `LeafyVGG-16_AdversarialV2.py` | Loads a trained model and evaluates clean-image performance and robustness against FGSM and PGD attacks. |

## Dataset

This work uses the publicly available **Tomato-Village** dataset:

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/mamtag/tomato-village?resource=download)

The dataset is provided by its original creators and remains subject to
its original license and attribution requirements.

### Directory Structure

Prepare separate training, validation, and test folders. Each split
must contain the same class subfolders:

```text
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Replace the example class names with your actual class names. Keep the
splits independent and avoid placing duplicate images in different splits.

The scripts determine class order by sorting the folder names in `train/`.
Use the same class names and ordering when evaluating a saved model.

## Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

A compatible GPU can accelerate training and adversarial evaluation.
An internet connection is required to download the ImageNet weights
on the first training run unless they are already cached.

## How to Run

### 1. Train LeafyVGG-16

Open `LeafyVGG-16_PretrainedV1.py` and change `base_path` to your dataset
directory:

```python
base_path = r"C:/path/to/dataset"
```

Run the training script:

```bash
python LeafyVGG-16_PretrainedV1.py
```

The script:

- Loads the training, validation, and test images.
- Applies augmentation to training images.
- Trains the classification head and fine-tunes the backbone.
- Saves model checkpoints and training history.
- Displays training and validation accuracy and loss curves.
- Reports test-set classification metrics.

### Training Outputs

Files are saved in the current working directory:

| Output | Description |
|---|---|
| `best_vgg16_250Epoch.keras` | Model checkpoint selected using validation accuracy. |
| `vgg16_final_model_250Epoch.keras` | Model saved after the training stages finish. |
| `vgg16_history_250Epoch.json` | Combined training and validation history from both stages. |

The configured maximum is 250 epochs across both stages. Early stopping
may reduce the actual number of epochs.

The checkpoint and final model can differ because checkpointing monitors
validation accuracy, while early stopping monitors validation loss.

### 2. Evaluate Adversarial Robustness

Open `LeafyVGG-16_AdversarialV2.py` and update:

1. `base_path` to the same dataset directory.
2. The model-loading path to your trained checkpoint.

For example:

```python
model = tf.keras.models.load_model(
    "best_vgg16_250Epoch.keras"
)
```

For consistency with the training script, set the test preprocessing to:

```python
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)
```

Run:

```bash
python LeafyVGG-16_AdversarialV2.py
```

The script loads the test images into memory, evaluates clean-image
performance, generates adversarial examples, and prints evaluation
results in the terminal.

## Adversarial Evaluation

The script implements **white-box, untargeted attacks** using the trained
model's gradients and the true class labels.

| Attack | Settings |
|---|---|
| FGSM | Epsilon values: 0.005, 0.01, 0.03, and 0.05 |
| PGD | Epsilon: 0.05; step size: 0.005; iterations: 10 |

FGSM applies a single gradient-sign perturbation. PGD applies repeated
gradient-sign updates and projects the perturbed input back into the
specified epsilon range around the original input.

This script evaluates adversarial robustness; it does not perform
adversarial training.

### Interpreting Attack Strength

In the supplied implementation, perturbations are applied directly to
preprocessed pixel values rather than to images normalized to `[0, 1]`.
Epsilon and step-size values must therefore be interpreted in those
pixel-intensity units.

PGD constrains the perturbation to the specified epsilon range. Neither
attack currently clips the resulting images to valid pixel-intensity
bounds. PGD starts from the original input without random initialization.

## Evaluation Metrics

Both scripts report:

- Accuracy.
- Macro-averaged precision.
- Macro-averaged recall.
- Macro-averaged F1-score.
- Per-class classification report.
- Confusion matrix.

The adversarial script reports these metrics for clean images and for
each attack setting.

## Paper

**LeafyVGG-16: Transfer Learning for Plant Disease Detection with Cyber Risk Analysis**

**Authors:** N. Chiwele, E. Sweeney, and K. Hossain.

**Status:** Accepted at the 2026 IEEE Global Humanitarian Technology
Conference (GHTC).

**Preprint:**  
https://www.biorxiv.org/content/10.64898/2026.05.13.724946v1

**DOI:**  
https://doi.org/10.64898/2026.05.13.724946

## Citation

If you use this code or build upon this work in your research, please
cite the associated paper:

> N. Chiwele, E. Sweeney, and K. Hossain, “LeafyVGG-16: Transfer Learning
> for Plant Disease Detection with Cyber Risk Analysis,” accepted at the
> 2026 IEEE Global Humanitarian Technology Conference (GHTC), 2026.
> Preprint available on bioRxiv.
> https://doi.org/10.64898/2026.05.13.724946

### BibTeX for the Preprint

```bibtex
@article{chiwele2026leafyvgg16,
  title   = {LeafyVGG-16: Transfer Learning for Plant Disease Detection with Cyber Risk Analysis},
  author  = {Chiwele, N. and Sweeney, E. and Hossain, K.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.13.724946},
  url     = {https://www.biorxiv.org/content/10.64898/2026.05.13.724946v1}
}
```

The citation will be updated with the IEEE proceedings details when
available.

## License

The source code is released under the **MIT License**. See the
`LICENSE` file for the full terms.

Third-party datasets and dependencies remain subject to their respective
licenses. The repository's MIT License does not change the license of
the Tomato-Village dataset or the associated paper.
