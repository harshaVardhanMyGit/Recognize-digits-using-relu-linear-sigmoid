# Recognize digits using ReLU, linear, and sigmoid

A small TensorFlow/Keras notebook project for handwritten digit recognition. It demonstrates activation functions, softmax, model construction, training, and prediction using a dataset of 20x20 grayscale digit images.

---

## Project structure

- digits_Prediction.ipynb — main notebook with the full workflow
- autils.py — helper plotting, visualization, data loading, and evaluation functions
- lab_utils_softmax.py — softmax visualization utility
- public_tests.py — simple checks for softmax and model architecture
- X.npy — input feature matrix (5000 examples × 400 pixels)
- y.npy — labels vector for the digit images
- deeplearning.mplstyle — matplotlib plotting style
- images — diagram and notebook image assets

---

## What this project does

- Visualizes ReLU, sigmoid, and linear activation functions
- Implements a NumPy `my_softmax` function
- Loads handwritten digit data from X.npy and y.npy
- Builds a Keras Sequential model with:
  - input shape `400`
  - hidden layer 1: `25` units, ReLU
  - hidden layer 2: `15` units, ReLU
  - output layer: `10` units, linear
- Compiles the model with:
  - `SparseCategoricalCrossentropy(from_logits=True)`
  - `Adam` optimizer
- Trains the model for `40` epochs
- Plots training loss
- Predicts digits and converts logits to probabilities using softmax
- Displays sample digits and model errors

---

## Requirements

- Python 3.x
- numpy
- matplotlib
- tensorflow
- ipywidgets (optional, for notebook widgets)
- Jupyter Notebook or VS Code Notebook support

---

## Installation

```bash
pip install numpy matplotlib tensorflow ipywidgets
```

---

## How to run

Open the notebook:

```bash
jupyter notebook digits_Prediction.ipynb
```

Or open digits_Prediction.ipynb directly in VS Code.

---

## Notebook workflow

1. Import packages and helper modules
2. Plot activation functions with `plt_act_trio()`
3. Implement softmax in `my_softmax(z)`
4. Load and inspect dataset shapes
5. Visualize sample digit images
6. Build the Keras model
7. Compile and train the model
8. Plot the loss curve with `plot_loss_tf(history)`
9. Predict a digit and convert logits to probabilities
10. Display sample prediction results and classification errors

---

## Notes

- The final layer uses `linear` activation to keep outputs as logits
- Softmax is applied separately when probabilities are needed
- Prediction labels are selected with `np.argmax`
- The dataset is mapped from `20×20` images to `400`-element vectors

---

## Useful helper functions

From autils.py:
- `load_data()`
- `plt_act_trio()`
- `display_digit()`
- `plot_loss_tf(history)`
- `display_errors(model, X, y)`

From lab_utils_softmax.py:
- `plt_softmax(my_softmax)`

From public_tests.py:
- `test_my_softmax(target)`
- `test_model(target, classes, input_size)`

---
