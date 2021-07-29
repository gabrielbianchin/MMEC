# MMEC: Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction

## Introduction

The protein secondary structure prediction is an important task with many applications, such as local folding analysis, tertiary structure prediction, and function classification. Driven by the recent success of multi-modal classifiers, new studies have been conducted using this type of method in other domains, for instance, biology and health care. In this work, we investigate the ensemble of three different classifiers for protein secondary structure prediction. Each classifier of our method deals with a transformation of the original data into a specific domain, such as image classification, natural language processing, and time series tasks. As a result, each classifier achieved competitive results compared to the literature, and the ensemble of the three different classifiers obtained 77.9% and 73.3% of Q8 accuracy on the CB6133 and CB513 datasets, surpassing state-of-the-art approaches in both scenarios.

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

## Citation

This repository contains the source codes of Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction, as given in the paper:

Gabriel Bianchin de Oliveira, Helio Pedrini, Zanoni Dias. "Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction", in proceedings of the 19th International Conference on Computer Analysis of Images and Patterns (CAIP). Virtual Conference, 27 September - 01 October 2021.

If you use this source code and/or its results, please cite our publication:

```
@inproceedings{Oliveira_2021_CAIP,
  author = {G.B. Oliveira and H. Pedrini and Z. Dias},
  title = {{Multi-Modal Ensemble Classifier for Protein Secondary Structure Prediction}},
  booktitle = {19th International Conference on Computer Analysis of Images and Patterns (CAIP)},
  address = {Virtual Conference},
  month = sep # "/" # oct,
  year = {2021}
}
```
