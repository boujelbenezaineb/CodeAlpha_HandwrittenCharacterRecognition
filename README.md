# Handwritten Character Recognition

A Convolutional Neural Network built with TensorFlow/Keras that recognizes handwritten alphabet characters from grayscale images, reaching **98% accuracy** on both training and validation sets.

Completed as **Task 3** of the Machine Learning internship at [CodeAlpha](https://www.linkedin.com/company/codealpha/).

---

## Overview

The goal of this project is to classify images of handwritten letters into their corresponding character classes. The pipeline covers the full workflow: loading and reshaping the raw dataset, exploring the class distribution, building a sequential CNN, training it, and visualizing predictions on unseen test images.

| | |
|---|---|
| **Task** | Multi-class image classification |
| **Framework** | TensorFlow / Keras |
| **Architecture** | Sequential CNN (Conv2D + MaxPooling) |
| **Optimizer** | Adam |
| **Epochs** | 20 |
| **Accuracy** | ~98% (training and validation) |

---

## Dataset

<!-- TODO: replace with the dataset you actually used and add the link -->
The model is trained on a dataset of handwritten alphabet images stored as flattened 28×28 grayscale pixel values, with one label column indicating the character class.

**Preprocessing steps:**

1. Load the dataset and separate features from labels.
2. Reshape each flattened row into a `28 × 28 × 1` image tensor suitable for the convolutional input layer.
3. Normalize pixel values to the `[0, 1]` range.
4. Split the data into training and validation sets.
5. Plot the distribution of samples per letter to check for class imbalance.

---

## Model Architecture

A sequential CNN that stacks convolutional blocks to progressively extract features, then flattens into dense layers for classification:

```
Input (28, 28, 1)
  ↓
Conv2D  → ReLU  → MaxPooling2D
  ↓
Conv2D  → ReLU  → MaxPooling2D
  ↓
Conv2D  → ReLU  → MaxPooling2D
  ↓
Flatten → Dense → ReLU
  ↓
Dense → Softmax (one unit per character class)
```

The convolutional layers learn stroke-level features such as edges, curves, and junctions, while the pooling layers reduce spatial dimensions and add a degree of translation invariance. The dense head maps the learned representation to class probabilities.

**Compilation:**

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Metric:** Accuracy

---

## Results

The model converges quickly and generalizes well, with training and validation accuracy tracking closely together across all 20 epochs — an indication that the network is not overfitting.

- **Training accuracy:** ~98%
- **Validation accuracy:** ~98%

Predictions are visualized on a grid of test images with the predicted character rendered alongside each sample, making the model's behavior easy to inspect qualitatively rather than relying on the accuracy number alone.

<!-- TODO: add screenshots
![Training curves](assets/accuracy_plot.png)
![Sample predictions](assets/predictions.png)
-->

---

## Getting Started

### Prerequisites

```
Python 3.8+
```

### Installation

```bash
git clone https://github.com/boujelbenezaineb/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, the core dependencies are:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
```

### Running the project

<!-- TODO: update to match your actual filenames -->
```bash
jupyter notebook handwritten_character_recognition.ipynb
```

Run the cells in order to reproduce preprocessing, training, and the prediction visualizations.

---

## Project Structure

<!-- TODO: update to match your repo -->
```
CodeAlpha_HandwrittenCharacterRecognition/
├── handwritten_character_recognition.ipynb   # Main notebook
├── data/                                     # Dataset
├── assets/                                   # Plots and screenshots
├── requirements.txt
└── README.md
```

---

## Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn`

---

## Key Takeaways

- Reshaping flattened pixel data into proper image tensors is essential before it reaches a convolutional layer.
- Stacking Conv2D and MaxPooling blocks builds up increasingly abstract feature representations with relatively few parameters.
- Visualizing predictions on real test samples reveals failure modes that a single accuracy score hides — particularly confusion between visually similar characters.

---

## Acknowledgments

Built during the Machine Learning internship at [CodeAlpha](https://www.linkedin.com/company/codealpha/).

---

## Author

**Zaineb Boujelbene**
GitHub: [@boujelbenezaineb](https://github.com/boujelbenezaineb)
