# Machine Learning Final Project: Customer Churn
Thompson Rivers University
COMP 4980 - Special Topics, Machine Learning
Bradley Crump

This project aims to show a student how to interact with data, formalize a plan for creating a predictive model and implement that model.

### Repository Information
Much of the project structure derives from Abishek Thakur's book, [Approaching Almost Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost).

Much of the functionality of the repo's code can be interacted with through the command line. The goal is to create an append-only, reusable codebase that can be applied to arbitrary use-cases. However, there are dependencies

A note: Run scripts from the root of the project
### Model
`Confusion Matrix`
Use confusion_matrix.py to produce a visualization of a confusion matrix.

Config Dependencies:
- INFERENCE_MODEL - model that will be used to create confusion matrix
- TRAINING_FILES - First file in the list will be used to retrieve the dataset

```python
python src/conufusion_matrix.py --fold <fold-number>
```

`Feature Importances`
Use feature_importance.py to produce a visualizaiton of the importance of features.

Config Dependencies:
- INFERENCE_MODEL - model that will be used to create confusion matrix
- TRAINING_FILES - First file in the list will be used to retrieve the dataset

```python
python src/feature_importance.py
```
