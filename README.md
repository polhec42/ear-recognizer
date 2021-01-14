# ear-recognizer

This repository contains all code that was written for an ear recognizer. Datasets are not included. Models are available at `https://drive.google.com/drive/folders/18tWs0tgIrBhwk6fgLN9-_mgu4WE9GE4e?usp=sharing` and are available to the people with access to the Univerza v Ljubljani. If there are any problems, feel free to contact me: `zm0971@student.uni-lj.si`.

In order to re-run the evaluation stage, run the `python evaluation.py`. This script will calculate the results for Rank-5 metric. 

## Directory structure

### classifier
File `classifier.py` contains PyTorch code that was used to design and train the CNN ear recognizer. 

### evaluate
File `evaluate.py` contains Python code that evaluates the model, that is present in the same directory with the name `best.pt`.

### prepare_dataset
File `prepare_dataset.py` was used to convert dataset in the correct form that is used for Pytorch CNN. 

### augmentor
File `augmentor.py` was used to augment the data with additional rotations and flips in order to obtain more data and consequently more robust classifer.

### gather_data
File `gather_data.py` was used to create `.csv` file that was used by `prepare_dataset.py` in order to correctly label data. This was only used in order to try with training the classifier that would predict gender/ethnicity, but then I pivoted to the identity recognition.

### plotter
File `plotter.py` contains Python code that was used to plot joint CMC curve for the report.


