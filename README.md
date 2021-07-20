# MMEC: Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction

## Introduction

This repository presents the codes of the paper "MMEC: Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction", presented at the 19th International Conference on Computer Analysis of Images and Patterns.

## Classifier

The protein secondary structure classifier presented in the paper is composed of an ensemble of Convolutional Neural Networks, BERT-based methods, and Inception Recurrent Networks. More details can be found in the paper.

## Reproducibility 

For CB6133:
```
1. Run all files in Convolutional Neural Network/CB6133
2. Run all files in BERT/CB6133
3. Run Inception Recurrent Networks/CB6133/IRN.py
4. Make the ensemble between CNNs using Genetic Algorithm/GA.py
5. Make the ensemble between BERTs using Genetic Algorithm/GA.py
6. Make the ensemble between IRNs using Genetic Algorithm/GA.py
7. Make the final ensemble between the classifiers using Genetic Algorithm/GA.py
```

For CB513:
```
1. Run all files in Convolutional Neural Network/CB513
2. Run all files in BERT/CB513
3. Run Inception Recurrent Networks/CB513/IRN.py
4. Make the ensemble between CNNs using Genetic Algorithm/GA.py
5. Make the ensemble between BERTs using Genetic Algorithm/GA.py
6. Make the ensemble between IRNs using Genetic Algorithm/GA.py
7. Make the final ensemble between the classifiers using Genetic Algorithm/GA.py
```