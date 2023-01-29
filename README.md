# Mathemathical Modeling Practice - Railway track defect detection

This project is intended as the project work for the class Mathemathical Modeling Practice under the specialization program of AI Research Group of Eötvös Lóránd University.

https://ai.elte.hu/training/

The target of the project is to work out a classification tool for detecting railway track defects.

The selected dataset that can be found in the Kaggle webpage.

https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection

## Summary

Five convolutaional neural networks built to classify the rail tracks. The applied networks and methodology are the following:

| Neural network type | Methodology        | Comment                    | ID         |
| ------------------- | ------------------ | -------------------------- | ---------- |
| LeNet-5             | pure model         |                            | LeNet-5    |
| AlexNet             | pure model         |                            | AlexNet    |
| VGG16               | pure model         |                            | VGG16      |
| VGG16               | transform learning |                            | VGG16_p    |
| ResNet50            | transform learning | Dense layers: 1024, 512, 1 | ResNet50_p |

Besides training the models, the learning rate is hypertuned and bootstrapping is applied for final test measure.

Best performance: VGG16 with transform learning

Major finding: All models behave as random classifier on the test data

![Results](/tex_graphs/bootstrap_results.png)

## Quickstart

The data is stored in different stages of the processing in the following folders:
- `./raw/`: Contains the raw dataset directly copied from Kaggle.
- `./data/`: The images after data cleaning in standard format and naming. (Created by the scripts.)
- `./augmented/`: Additional images as result of data augmentation. (Created by the scripts.)
- `./preprocessed/`: The data after image processing, preprocessed for the machine learning
	    algorithm, includes both the original and augmented dataset. (Created by the scripts.)
- `./models/`: Stores the trained classification models. (Created by the scripts.)

The code structure is as follows:
- `./data_cleaning.ipynb`: Processes data from `raw` to `data` state.
- `./data_explorer.ipynb`: Contains the data exploration steps.
- `./classification_model_*.ipynb`: The overall CNN based classification model,
	    where * stands for the applied network model.
- `./playground.ipynb`: Contains the algorithms that are under development.
- `./toolkit/classes.py`: Stores the class definitions
- `./hp_tuner.ipynb`: Tool used for hyperparameter tuning.
- `./logs/`: Log files created during the training of neural networks. (Created by the scripts.)
- `./tuner/`: Log and model files created by the hypertuner. (Created by the scripts.)

The progress and results are [documented](build/documentation.pdf) in the `./documentation.tex` file, a pdf format can be found in `./build/`
The corresponding folders and files:
- `./documentation.tex`: $\LaTeX$ documentation
- `./presentation.tex`: Final presentation of the project
- `./tex_graphs/`: To store the images inserted into the $\LaTeX$ document
- `./tex_refs/`: Reference files for the \LaTeX documentation
- `./build/`: Temporary build directory for the $\LaTeX$ file, contains the pdf as default

