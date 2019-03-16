# semi-markov-crf

Code for paper *Neural Semi-Markov Conditional Random Fields for Robust Character-Based Part-of-Speech Tagging* \[IN PROGRESS\]

## Requirements
python 3.6
pytorch 0.2.0
numpy
sklearn

#Create a new dataset (Optional)

Create a new char datasets from by specifying the word files (input) and char files(out) and running data_loader.py
The word files must follow the UD format

##Train

Go to train.py from arg\_list select the model\_path and the LANG (language, currently 'en' for English  UD 1.2 or 'vi' for Vietnamese UD 2.0)

To train the model with the default parameters now run `python3 train.py`

Currently train on CPU, GPU might cause a slow start problem due to CUDA version and PyTorch version missmatch bug (looking into it, runned ok on cis lambda)

Default training on Vietnamese which is a smaller dataset (training takes 1 day vs 3 days on english)

##Evaluate

After training a model go to predict.py and select the model\_path and the LANG you want to evaluate

Repo has currently an untrained model on Vietnamese





