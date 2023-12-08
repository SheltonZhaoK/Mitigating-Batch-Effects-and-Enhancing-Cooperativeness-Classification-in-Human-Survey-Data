import warnings, os, argparse, json
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import urllib.request as ur
import xgboost as xgb

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data import (encode_label, transform_2_oneWavePsample, transform_2_allWavePsample, make_data)
from configs import Configs
from utilities import create_mlp_args_adatively, format_gridSearch_results, make_dir, output_feature_importance, set_seed
from plot import preliminary_scatter, augmentation_scatter

def main(configs, args, report):
    ################################### read data ###################################
    data = pd.read_csv(os.path.join(configs["inputDir"], "ICPSR_36371/DS0001/36371-0001-Data.tsv"), sep="\t", na_values=[" "])
    label = "W3IWER4_A" # How Cooperative Was Respondent? (Iwer's assessment)
    if args.e == "preliminary":
        data = data.drop(columns = ["IWER4_A", "W3IWER4_B", "IWER4_B"])
        target = data[["RWAVENEW", "RWAVEOLD"]]
    else:
        data = data.drop(columns = ["IWER4_A", "W3IWER4_B", "IWER4_B","RWAVENEW", "RWAVEOLD"])
    
    # label = "W3IWER4_B" # How Interested Was Respondent? (Iwer's assessment)
    # data = data.drop(columns = ["IWER4_A", "W3IWER4_A", "IWER4_B"])
    
    if args.t is not None:
        if args.t == "oneW":
            data = transform_2_oneWavePsample(data)
        elif args.t == "allW":
            data = transform_2_allWavePsample(data)
    data = encode_label(data, label)
  
    ################################### stratified train test split ###################################
    train_data, test_data = train_test_split(data, test_size=configs["test_size"], random_state=configs["seed"], stratify=data[label])
    print(f"Training subset: {len(train_data)}, Testing subset: {len(test_data)}")

    # -------------------------------------------------------- training data --------------------------------------------------------
    train_data, train_labels, dataCleaner, feature_selector, pc_reducer, umap_reducer = make_data(train_data, label, configs, args, type = "training")
    outputTrain = train_data.copy()
    outputTrain[label] = train_labels[label]
    outputTrain.to_csv(os.path.join(configs["outputDir"], args.e, f"A{args.a}_D{args.d}_T{args.t}_G{args.gan}_trainingData.csv"))

    assert len(train_labels) == len(train_data), "size of data and label does not match"
    assert train_labels[label].value_counts().std() == 0, "training data is not balanced" 

    if args.e == "preliminary":
        preliminary_scatter(train_data, train_labels, label, target, args, configs)
    if args.a is not None:
        augmentation_scatter(train_data, train_labels, label, args, configs)
    # -------------------------------------------------------- testing data --------------------------------------------------------
    test_data, test_labels = make_data(test_data, label, configs, args, "testing", dataCleaner, feature_selector, pc_reducer, umap_reducer)
    assert len(test_labels) == len(test_data), "size of data and label does not match"
    assert len(test_data.columns) == len(train_data.columns), "size of features of training and testing does not match"
    predictions = test_data.copy()
    predictions["label"] = test_labels[label]
    classifiers = {
                    "naive": GridSearchCV(DummyClassifier(), param_grid=configs["Classifiers"]["naive"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "MLP": GridSearchCV(MLPClassifier(), param_grid=create_mlp_args_adatively(configs, len(data), numLayers = 3), cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "logisticRegression": GridSearchCV(LogisticRegression(), param_grid=configs["Classifiers"]["logisticRegression"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "decisionTree": GridSearchCV(DecisionTreeClassifier(), param_grid=configs["Classifiers"]["decisionTree"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "randomForest": GridSearchCV(RandomForestClassifier(), param_grid=configs["Classifiers"]["randomForest"], cv=5, scoring="accuracy", n_jobs = -1, refit=True),
                    "xgb": GridSearchCV(xgb.XGBClassifier(enable_categorical = True), param_grid=configs["Classifiers"]["xgb"], cv=5, scoring="accuracy", n_jobs = -1, refit=True)
                  }
    
    for name in classifiers:
        print(f"{'+'*40} Tune and fit {name} {'+'*40}")
        classifier = classifiers[name]
        classifier.fit(train_data.values, train_labels.values.ravel())
        format_gridSearch_results(classifier, os.path.join(configs["outputDir"], "hyperparameter_tuning" ,f"{name}_gridSearch.txt"))
        train_accuracy = round(accuracy_score(classifier.predict(train_data), train_labels), 4)
        print(f"{name} train accuracy: {train_accuracy}")

        if name in ["decisionTree", "randomForest", "xgb"]:
            output_feature_importance(name, train_data, classifier, args, configs)

        print(f"{'+'*40} Test {name} with test data {'+'*40}")
        prediction = classifier.predict(test_data.values)
        predictions[name] = prediction
        test_accuracy = round(accuracy_score(prediction, test_labels), 4)
        test_wF1 = round(f1_score(prediction, test_labels, average="weighted"),4)
        test_cohen = round(cohen_kappa_score(prediction, test_labels),4)
        print(f"{name} test accuracy: {test_accuracy}")
        report.loc[len(report)] = [args.d, args.t, f"{train_data.shape}", f"{test_data.shape}",
                                   name, train_accuracy, test_accuracy, test_wF1, test_cohen, args.a, args.gan, configs["seed"]]
        predictions.to_csv(os.path.join(configs["outputDir"], args.e, f"A{args.a}_D{args.d}_T{args.t}_G{args.gan}_testingData_predications.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-t', choices = ["oneW", "allW"], type=str, default=None, help='specify two type of transformations to handle the original data')
    parser.add_argument('-d', choices = ["raw", "pca", "umap", "pca_umap"], type=str, default=None, help='specify three type of data to use')
    parser.add_argument('-a', type=str, default=None, help='choose which data augmentation method to use')
    parser.add_argument('-e', required = True, choices = ["explore", "preliminary", "baseline", "augmentation"], type=str, default=None, help='experiments name')
    parser.add_argument('-gan', action="store_true", help='whether to use GAN to increase the size of training data')
    args = parser.parse_args()

    report = pd.DataFrame(columns = ["Data Type", "Data Transformation", "X_train Size", "X_test Size", "Classifiers", "Training Accuracy", "Accuracy", 
                                     "Weighted F1", "Cohen Kappa", "Data Augmentation Methods", "GAN", "seed"])
    if args.e == "explore":
        assert args.d is not None, "-d must be specified for exploration"
        if args.gan:
            assert args.a is not None, "GAN must be used with the presence of the augmentation method"
        configs = Configs().configs
        set_seed(configs["seed"])
        main(configs, args, report)
        report.to_csv(os.path.join(make_dir(os.path.join(configs["outputDir"], args.e)) , f"D{args.d}_T{args.t}_A{args.a}_results.csv"))

    elif args.e == "preliminary":
        assert args.t is None, "-t must be None for preliminary experiment"
        assert args.a is None, "-a preliminary experiment does not need data augmentation method"
        for seed in [1]:
            configs = Configs(seed).configs
            set_seed(configs["seed"])
            for dataType in ["raw", "pca", "umap", "pca_umap"]:
                outputDir = make_dir(os.path.join(configs["outputDir"], args.e))
                args.d = dataType
                main(configs, args, report)
        report.to_csv(os.path.join(outputDir , f"{args.e}_results.csv"))

    elif args.e == "baseline":
        for seed in [1]:
            configs = Configs(seed).configs
            set_seed(configs["seed"])
            outputDir = make_dir(os.path.join(configs["outputDir"], args.e))
            for transformation in [None, "oneW"]:
                for dataType in ["raw", "pca", "umap", "pca_umap"]:
                    args.d = dataType
                    args.t = transformation
                    print(args, seed)
                    main(configs, args, report)
                    print(report)
        report.to_csv(os.path.join(outputDir , f"{args.e}_results.csv"))
    
    elif args.e == "augmentation":
        for seed in [1]:
            configs = Configs(seed).configs
            set_seed(configs["seed"])
            outputDir = make_dir(os.path.join(configs["outputDir"], args.e))
            for dataType in ["raw", "pca", "umap", "pca_umap"]:
                for transformation in [None, "oneW"]:
                    for augmentation in ["smote", "editNN", "tomkLink", "smoteNN", "smoteTomek"]:
                        for GAN in [False, True]:
                            args.t = transformation
                            args.d = dataType
                            args.a = augmentation
                            args.gan = GAN
                            print(args, seed)
                            main(configs, args, report)
        report.to_csv(os.path.join(outputDir, f"{args.e}_results.csv"))
                            
                    