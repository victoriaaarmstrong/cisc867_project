# CISC 867 Final Project Code

This repository contains the code for the CISC 856 Final Project. 

- _model.py_ contains the generator and discriminator models as well as the training function
- _dataset.py_ contains the functions for overwriting the VisionDataset class to make a custom Dataloader that stores two images as pairs (real, sparse)
- _testing.py_ contains code used to test the model after it had been trained 
- _participantdata.py_ contains the code that created the dataset from the [WESAD Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) 

Given the size of the dataset, I couldn't include all of the data used but I've included one batch of images as well as three checkpoints (beginning, midway, end) in case you would like to test anything out. 

The Frechet Inception Distance was calculated using [pytorch_fid](https://github.com/mseitzer/pytorch-fid)

The base code structure for the DCGAN architecture came from the [Pytorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
