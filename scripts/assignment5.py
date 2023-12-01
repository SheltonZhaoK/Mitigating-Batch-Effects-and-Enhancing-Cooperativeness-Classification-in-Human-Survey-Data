import warnings, os, argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as ur


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
    data, labels, dataCleaner, feature_selector = make_data(data, label, configs, args, type = "training")

    assert len(labels) == len(data), "size of data and label does not match"
    assert labels[label].value_counts().std() == 0, "training data is not balanced"

    # -------------------------------------------------------- testing data --------------------------------------------------------
    # data auditing
    data = test_data
    data, labels = make_data(data, label, configs, args, "testing", dataCleaner, feature_selector)
    assert len(labels) == len(data), "size of data and label does not match"

if __name__ == "__main__":
    configs = Configs().configs
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-t', type=str, default=None, help='specify two type of transformations to handle the original data (oneW, allW)')
    parser.add_argument('-d', type=str, default=None, help='specify three type of data to use')
    args = parser.parse_args()
    print(configs)
    main(configs, args)