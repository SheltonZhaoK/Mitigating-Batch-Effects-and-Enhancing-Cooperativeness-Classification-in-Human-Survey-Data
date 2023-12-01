# python3 assignment5.py -f both (munal + other two) #W3CL1R
import warnings, os, argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as ur

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from data import (NaColumnsHandler, NaHandler, duplicateHandler, rowOutlierChecker, correlatedFeatureRemover, 
#                         normalizer, variableFeaturesSelector, pcaReducer, audit_data, umapReducer, impute_NaN,
#                         select_features_ttest, select_features_importance, select_features_manual, select_by_wave, transform_2_allWavePsample, transform_2_oneWavePsample,
#                         reduce_data)

from data import *

from configs import Configs

def main(configs, args):
    ################################### read data ###################################
    data = pd.read_csv(os.path.join(configs["inputDir"], "ICPSR_36371/DS0001/36371-0001-Data.tsv"), sep="\t", na_values=[" "])
    # label = "W3IWER4_A" # How Cooperative Was Respondent? (Iwer's assessment)
    # data = data.drop(columns = ["IWER4_A", "W3IWER4_B", "IWER4_B"])
    
    label = "W3IWER4_B" # How Interested Was Respondent? (Iwer's assessment)
    data = data.drop(columns = ["IWER4_A", "W3IWER4_A", "IWER4_B"])
    
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
    # data auditing
    data = train_data
    report = audit_data(data, duplicateRow = configs["DataProcessing"]["audit"]["duplicateRow"], duplicateCol = configs["DataProcessing"]["audit"]["duplicateCol"], 
                        NaN = configs["DataProcessing"]["audit"]["NaN"], column_NaN = configs["DataProcessing"]["audit"]["column_NaN"],  
                        maxNA = configs["DataProcessing"]["audit"]["maxNA"])
    
    # data cleaning
    print(f"{'*'*30} Data Cleaning {'*'*30}")
    dataCleaner = Pipeline([
                            ("NaColumnsHandler", NaColumnsHandler(report=report)),
                            ("NaHandler", NaHandler(report=report, evaluation=configs["DataProcessing"]["cleaning"]["evaluation"])),
                            ("duplicateHandler", duplicateHandler(report=report, evaluation=configs["DataProcessing"]["cleaning"]["evaluation"])),
                            ("rowOutlierChecker", rowOutlierChecker(stdScale=configs["DataProcessing"]["cleaning"]["stdScale"], evaluation=configs["DataProcessing"]["cleaning"]["evaluation"]))
                            ])
    data = dataCleaner.fit_transform(data)
    target = data[[label]]
    data = data.drop(columns=[label])
    # features processing
    print(f"{'*'*30} Select Features {'*'*30}")
    feature_selector = Pipeline([
                        ("dataBalancer", dataBalancer(metaData = target, column2balance = label, training = True, outputDir = "../output")),
                        ("correlatedFeatureRemover", correlatedFeatureRemover(upperBound=configs["DataProcessing"]["feature_selection"]["correlationUpperBound"])),
                        ("normalizer", normalizer(lowerBound=configs["DataProcessing"]["feature_selection"]["scaleRange"][0],upperBound=configs["DataProcessing"]["feature_selection"]["scaleRange"][1])), 
                        ("variableFeaturesSelector", variableFeaturesSelector(numFeatures=configs["DataProcessing"]["feature_selection"]["numVariableFeatures"]))
                        ])
    
    training_indices = np.load(os.path.join(configs["outputDir"], 'training_indices.npy'))
    pc_reducer = Pipeline([("pca", pcaReducer(criteria=configs["DataProcessing"]["feature_selection"]["pcaCriteria"]))])
    umap_reducer = Pipeline([("umap", umapReducer(seed=configs["seed"]))])

    data = feature_selector.fit_transform(data)
    data.index = training_indices
    target = target.loc[training_indices, :]
    if args.d is not None:
        data = reduce_data(data, pc_reducer, umap_reducer, args)

    assert len(target) == len(data), "size of data and label does not match"
    assert target[label].value_counts().std() == 0, "training data is not balanced"

    # -------------------------------------------------------- testing data --------------------------------------------------------
    # data auditing
    data = test_data
    report = audit_data(data, duplicateRow = configs["DataProcessing"]["audit"]["duplicateRow"], duplicateCol = configs["DataProcessing"]["audit"]["duplicateCol"], 
                        NaN = configs["DataProcessing"]["audit"]["NaN"], column_NaN = configs["DataProcessing"]["audit"]["column_NaN"],  
                        maxNA = configs["DataProcessing"]["audit"]["maxNA"])
    
    # data cleaning
    print(f"{'*'*30} Data Cleaning {'*'*30}")
    dataCleaner = Pipeline([
                            ("NaColumnsHandler", NaColumnsHandler(report=report)),
                            ("NaHandler", NaHandler(report=report, evaluation=True)),
                            ("duplicateHandler", duplicateHandler(report=report, evaluation=True)),
                            ("rowOutlierChecker", rowOutlierChecker(stdScale=configs["DataProcessing"]["cleaning"]["stdScale"], evaluation=True))
                            ])
    data = dataCleaner.fit_transform(data)

    target = data[[label]]
    data = data.drop(columns=[label])

    print(f"{'*'*30} Select Features {'*'*30}")
    # features processing
    feature_selector.set_params(dataBalancer__training=False)
    data = feature_selector.transform(data)
    if args.d is not None:
        data = reduce_data(data, pc_reducer, umap_reducer, args)

    assert len(target) == len(data), "size of data and label does not match"

if __name__ == "__main__":
    configs = Configs().configs
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-t', type=str, default=None, help='specify two type of transformations to handle the original data (oneW, allW)')
    parser.add_argument('-d', type=str, default=None, help='specify three type of data to use')
    args = parser.parse_args()
    print(configs)
    main(configs, args)