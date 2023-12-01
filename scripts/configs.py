# -----------------------------------------------------------
# This script provides hyperparameter used in the experiments
#
# Author: Konghao Zhao
# Created: 2023-10-27
# Modified: 2023-10-30
# 
# -----------------------------------------------------------
class Configs:
    def __init__(self, seed = 1, max_iter = 400):
        self.configs = {
            "seed": seed,
            "test_size": 0.3,
            "inputDir": "../data",
            "outputDir": "../output",

            "DataProcessing":
            {
                "audit":
                {
                    "duplicateRow": True,
                    "duplicateCol": True,
                    "NaN": True,
                    "column_NaN": True,
                    "maxNA": 0.30 #0.25 -> 0-1265, 1-588, features-27 | 0.30 -> 0-1241, 1-563, features-62
                },

                "cleaning":
                {
                    "stdScale": 3,
                    "evaluation": False
                },

                "feature_selection":
                {
                    "correlationUpperBound": 0.98,
                    "scaleRange": [-1, 1],
                    "numVariableFeatures": 100,
                    # "pcaCriteria":0.98
                    "pcaCriteria":10
                },
            },

            "Classifiers":
            {
                "mlp":
                {
                    "solver": ["adam", "sgd"],
                    "alpha": [0.0001, 0.001],
                    "learning_rate": ["constant", "adaptive"],
                    "learning_rate_init": [0.001, 0.01],
                    "random_state": [seed],
                    "early_stopping": [True],
                    "validation_fraction": [0.2],
                    "max_iter": [max_iter]
                },

                "logisticRegression":
                {
                    "penalty": ["l1", "l2", "elasticnet"],
                    "C": [0.1, 1, 10],
                    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    "max_iter": [max_iter],
                    "random_state": [seed]
                },

                "decisionTree":
                {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # "max_depth": [2, 3, 5, 10, 20],
                    "min_samples_leaf": [5, 10, 20, 50, 100],
                    "random_state": [seed]

                },
                
                "randomForest":
                {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # "max_depth": [2, 3, 5, 10, 20],
                    'max_features': ["sqrt", "log2"],
                    "min_samples_leaf": [5, 10, 20, 50, 100],
                    "random_state": [seed]
                },

                "naive":
                {
                    "random_state": [seed]
                },

            }
        }