# Mathemathical Modeling Practice - Railway track defect detection

This project is intended as the project work for the class Mathemathical Modeling Practice under the specialization program of AI Research Group of Eötvös Lóránd University.

https://ai.elte.hu/training/

The target of the project is to work out a classification tool for detecting railway track defects.

The selected dataset that can be found in the Kaggle webpage.

https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection

## Summary

Five convolutaional neural networks built to classify the rail tracks. The applied networks and methodology are the following:

| Neural network type | Methodology        | Comment                    |
| ------------------- | ------------------ | -------------------------- |
| LeNet-5             | pure model         |                            |
| AlexNet             | pure model         |                            |
| VGG16               | pure model         |                            |
| VGG16               | transform learning |                            |
| ResNet50            | transform learning | Dense layers: 1024, 512, 1 |

Besides training the models, the learning rate is hypertuned and bootstrapping is applied for final test measure.

Best performance: VGG16 with transform learning
Major finding: All models result in random classifier for the test data

![Results](/tex_graphs/bootstrap_results.png)

## Quickstart

The data is stored in different stages of the processing in the following folders:
- `./raw/`: Raw dataset copied from Kaggle
- `./data/`: Cleaned dataset in standardized format (created by the scriptfiles)
- `./augmented/`: Dataset of augmented images (created by the scriptfiles)
- `./preprocessed/`: Images preprocessed for feeding into the neural network (created by the scriptfiles)

The code structure is as follows:
- `./toolkit/classes.py`: Stores the class definitions
- `./data_cleaning.ipynb`: Executes the raw data processing
- `./data_explorer.ipynb`: Contains the scripts for exploring the dataset
- `./classification_model_*.ipynb`: Invokes the construction, training and evaluation of the neural network, * stands for the network model
- `./playground.ipynb`: Contains the scripts under development or trial
- `./models/`: Stores saved models (created by the scripts)

The progress and results are documented in the `./documentation.tex` file, a pdf format can be found in `./build/`
The corresponding folders and files:
- `./build/`: Temporary build directory for the $\LaTeX$ file, contains the pdf as default
- `./tex_graphs/`: To store the images inserted into the $\LaTeX$ document
- `./tex_refs/bibliography.bib`: Bibliography
- `./tex_refs/style.sty`: Style file for $\LaTeX$
- `./documentation.tex`: $\LaTeX$ documentation

## The project is still in work.
