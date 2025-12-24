# Simple-ResNet-Classifier
A simple Residual Network for classifying generated shapes. 
Intended to simulate tumor classification in the liver.
Created as a mini-project for Dr. George Biros.

# Data generation
To simulate liver scans for our purposes, we randomly draw white ellipses
to represent the liver. For a proportion of these images, we randomly draw
a black ellipse inside of the white ellipse to represent the tumor.

We use `data/generation.py` to generate our dataset for our model. The `--dir`
flag sets the name of the directory for the images, which has the labels saved
in the `data/labels` directory.

To run:
```bash
python3 -m data.generation --dir train
```

Utilize the numerous flags defined in `data/generation.py` to paramaterize
the image dataset.

# Model definition and training
`model/model.py` defines a basic residual network and `model/train.py` defines
the training loop for training this model. The trained weights for this model
are stored in the same directory as `model/resnet.th`. The `model/logs` directory
stores past instances of weights.

To run:
```bash
python3 -m model.train
```

Utilize the numerous flags defined in `model/train.py` to paramaterize
the model and training code.

# Acknowledgements
- The model architecture and training loop structure were informed by the curriculum 
from CS 342, taught at UT Austin.
- Usage of Copilot for minor debugging.