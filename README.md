# Mathemathical-Modeling-Practice

This project is intended as the project work for the class Mathemathical Modeling Practice under the specialization program of AI Research Group of Eötvös Lóránd University.
https://ai.elte.hu/training/

The target of the project is to work out a classification tool for the selected dataset that can be found in the Kaggle webpage.
https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection

## Quickstart

The data is stored in different stages of the processing in the following folders:
- `raw`: Raw dataset copied from Kaggle
- `data`: Cleaned dataset in standardized format
- `augmented`: Dataset of augmented images
- `preprocessed`: Images preprocessed for feeding into the neural network

The code structure is as follows:
- `./toolkit/classes.py`: Stores the class definitions
- `./data_cleaning.ipynb`: Executes the raw data processing
- `./data_explorer.ipynb`: Contains the scripts for exploring the dataset
- `./classification_model.ipynb`: Invokes the construction, training and evaluation of the neural network
- `./playground.ipynb`: Contains the scripts under development or trial

The progress and results are documented in the `documentation.tex` file, a pdf format can be found in `./build/`

## The project is still in work.
