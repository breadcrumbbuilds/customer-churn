# config.py

KFOLD_INPUT = "./input/raw/customer_churn.csv"

TRAINING_FILES = [
    "input/processed/customer_churn.csv"
]

INFERENCE_MODEL = "models/run__2021_04_12_16_43/churn_decision-tree-gini_None_random-os_0.bin"

FEATURES = [
    # "InternetService_No",
    "TotalCharges",
    "MonthlyCharges",
    "tenure",
    # "InternetService_DSL",
    "target"
]



MODEL_OUTPUT = "models/"

"""     Random Forest and ExtraTrees     """
# GRID_SEARCH_PARAMS = {
#     "n_estimators": [500, 1000, 2500],
#     "max_depth": [3, 5, 12],
#     "max_features": [0.1, 'auto'],
#     "criterion": ["gini", "entropy"],
#     "bootstrap": [True, False]
# }


"""     Voting Classifier   """
GRID_SEARCH_PARAMS = {
    "et__n_estimators": [100, 1000],
    "gb__n_estimators": [100, 1000],
    "rf__n_estimators": [100, 1000],
    "ab__n_estimators": [100, 1000],

    "et__max_depth": [3, 30],
    "gb__max_depth": [3, 16],
    "rf__max_depth": [3, 7],

    "et__max_features":[.1, .5,],
    "gb__max_features":[.1,],
    "rf__max_features":[.1, .3],
    "et__bootstrap":[False, False],
    # "gb__bootstrap":[True],
    "rf__bootstrap":[True, False],
    "voting": ["hard", "soft"]
}

"""     Adaboost and GradientBoost   """
# GRID_SEARCH_PARAMS = {
#     "n_estimators": [100, 333, 1000, 3333, 10000],
#     "learning_rate": [.1,.3,1, 3, 10],
# }

"""     MLP     """
# GRID_SEARCH_PARAMS = {
#     "learning_rate_init": [0.01],
#     "random_state": [42],
#     "learning_rate" : ["adaptive"],
#     "alpha": [0.0001, 0.000001],
#     "batch_size": [32, 64, 128],
#     "solver": ["sgd", "adam"],
#     "activation": ["relu"],
#     "hidden_layer_sizes": [(1000,1000), (100,100,100,100)],
#     "n_iter_no_change": [10]
# }
