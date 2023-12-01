import warnings, os, argparse
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as ur

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data import (encode_label, transform_2_oneWavePsample, transform_2_allWavePsample, make_data)
from configs import Configs
from utilities import create_mlp_args_adatively, format_gridSearch_results

def main(configs, args):
    ################################### read data ###################################
    data = pd.read_csv(os.path.join(configs["inputDir"], "ICPSR_36371/DS0001/36371-0001-Data.tsv"), sep="\t", na_values=[" "])
    label = "W3IWER4_A" # How Cooperative Was Respondent? (Iwer's assessment)
    data = data.drop(columns = ["IWER4_A", "W3IWER4_B", "IWER4_B", "RWAVENEW", "RWAVEOLD"])
    # data = data.drop(columns = ["IWER4_A", "W3IWER4_B", "IWER4_B"])
    #  "RWAVENEW", "RWAVEOLD"***********
    
    # label = "W3IWER4_B" # How Interested Was Respondent? (Iwer's assessment)
    # data = data.drop(columns = ["IWER4_A", "W3IWER4_A", "IWER4_B"])
    
    data = encode_label(data, label)
    
    if args.t is not None:
        if args.t == "oneW":
            data = transform_2_oneWavePsample(data)
        elif args.t == "allW":
            data = transform_2_allWavePsample(data)
       
    ################################### stratified train test split ###################################
    train_data, test_data = train_test_split(data, test_size=configs["test_size"], random_state=configs["seed"], stratify=data[label])
    print(f"Training subset: {len(train_data)}, Testing subset: {len(test_data)}")

    # -------------------------------------------------------- training data --------------------------------------------------------
    data = train_data
    data, labels, dataCleaner, feature_selector, pc_reducer, umap_reducer = make_data(data, label, configs, args, type = "training")
    assert len(labels) == len(data), "size of data and label does not match"
    assert labels[label].value_counts().std() == 0, "training data is not balanced"

    classifiers = {
                    "naive": GridSearchCV(DummyClassifier(), param_grid=configs["Classifiers"]["naive"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "MLP": GridSearchCV(MLPClassifier(), param_grid=create_mlp_args_adatively(configs, len(data), numLayers = 3), cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "logisticRegression": GridSearchCV(LogisticRegression(), param_grid=configs["Classifiers"]["logisticRegression"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "decisionTree": GridSearchCV(DecisionTreeClassifier(), param_grid=configs["Classifiers"]["decisionTree"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "randomForest": GridSearchCV(RandomForestClassifier(), param_grid=configs["Classifiers"]["randomForest"], cv=5, scoring="accuracy", n_jobs = -1, refit=True)
                  }
    
    for name in classifiers:
        print(f"{'+'*40} Tune and fit {name} {'+'*40}")
        classifier = classifiers[name]
        classifier.fit(data, labels.values.ravel())
        format_gridSearch_results(classifier, os.path.join(configs["outputDir"], f"{name}_gridSearch.txt"))
        print(f"{name} train accuracy: {accuracy_score(classifier.predict(data), labels)}")

        if name in ["decisionTree", "randomForest"]:
            best_estimator = classifier.best_estimator_
            feature_importances = best_estimator.feature_importances_
            feature_importance_dict = dict(zip(data.columns, feature_importances))
            sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            print(f"{name} Feature Importances: {dict(sorted_feature_importance)}")

    # -------------------------------------------------------- testing data --------------------------------------------------------
    data = test_data
    data, labels = make_data(data, label, configs, args, "testing", dataCleaner, feature_selector, pc_reducer, umap_reducer)
    assert len(labels) == len(data), "size of data and label does not match"
    for name in classifiers:
        print(f"{'+'*40} Test the model with test data {name} {'+'*40}")
        classifier = classifiers[name]
        print(f"{name} test accuracy: {accuracy_score(classifier.predict(data), labels)}")

if __name__ == "__main__":
    configs = Configs().configs
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-t', choices = ["oneW", "allW"], type=str, default=None, help='specify two type of transformations to handle the original data (oneW, allW)')
    parser.add_argument('-d', required = True, choices = ["raw", "pca", "umap", "pca_umap"], type=str, default=None, help='specify three type of data to use')
    args = parser.parse_args()
    print(configs)
    main(configs, args)