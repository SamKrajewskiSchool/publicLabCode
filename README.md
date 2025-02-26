# publicLabCode
Note: THIS IS MEANT TO BE A CODE REFERENCE, NOT A STANDALONE SOFTWARE.

### Background
This repository contains a portion of the code developed by Sam Krajewski in the Speech Processing and Auditory Neuroscience Lab at UW - Madison. This code changes frequently and is not guaranteed to function as described. This code is converted from a messy jupyter notebook to a python file so it will not run by default. 

This code is meant to take Frequency Following Response (FFR) electroencaphalogram (EEG) data, preprocess it as required for the specific training instance, and use than data to train a Multilayered Perceptron (MLP) classification model. This is implemented using the PyTorch machine learning library. 

I am also actively investigating cutting-edge signal processing techniques for electroencephalogram data. Most recently I have been investigating the affects of Variation Model Decomposition (VMD) on the classification accuracies of our model across multiple instances.
