## Code Files

- **`LeafyVGG-16_PretrainedV1.py`**  
  Trains an ImageNet-pretrained VGG16 model with a custom classification
  head. Training includes up to 50 epochs with the VGG16 backbone frozen,
  followed by up to 200 epochs of fine-tuning. The script saves the model
  and training history, displays training curves, and evaluates performance
  on the test set.

- **`LeafyVGG-16_AdversarialV2.py`**  
  Loads the trained model and evaluates its performance on clean test images
  and adversarial images generated using FGSM and PGD. It reports accuracy,
  macro precision, macro recall, macro F1-score, classification reports,
  and confusion matrices.

## Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib scikit-learn
