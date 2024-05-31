# Plant Disease Identification

## Introduction
This repository analyzes the efficacy of deep learning models in classifying 14 plant species and detecting 20 diseases, including healthy specimens. It compares unified and separate model approaches, providing insights for automated plant disease diagnosis and species classification.

## Discussion
ResNet architectures outperform AlexNet models, highlighting the importance of residual connections for learning complex features. Disease identification is more challenging than species identification due to the larger number of classes and symptom similarities.

## Conclusions
ResNet models show superior performance in plant classification tasks. Disease identification is more complex than species classification. Future research should explore the impact of sample size on model overfitting to develop robust solutions.

## Model Performance

| Model      | Problem Type   | Accuracy | PR-AUC  |
|------------|----------------|----------|---------|
| ResNet-18  | Joint Problem  | 0.9624   | 0.9756  |
| ResNet-34  | Joint Problem  | 0.9586   | 0.9847  |
| AlexNet    | Species        | 0.9300   | 0.9285  |
| AlexNet    | Disease        | 0.8872   | 0.8550  |

## Repository Structure

```
├───all_classes_at_once
├───model_by_problem
└───Report
    └───images
```

Explore the repository for insights into plant disease identification and species classification.