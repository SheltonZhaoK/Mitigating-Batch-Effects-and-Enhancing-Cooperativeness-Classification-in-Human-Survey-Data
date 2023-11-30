# python3 assignment5.py -f both (munal + other two) #W3CL1R
import warnings, os, argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as ur

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dataObject import (NaColumnsHandler, NaHandler, duplicateHandler, rowOutlierChecker, correlatedFeatureRemover, 
                        normalizer, variableFeaturesSelector, pcaReducer, audit_data, umapReducer, impute_NaN,
                        select_features_ttest, select_features_importance, select_features_manual, select_by_wave, transform_2_allWavePsample, transform_2_oneWavePsample)

from configs import Configs

def main(configs, args):
    ################################### read data ###################################
    # label =  "CL1R" 
    label = "W3CL1R"
    data = pd.read_csv("../data/ICPSR_36371/DS0001/36371-0001-Data.tsv", sep="\t", na_values=[" "])
    
    if args.s is not None:
        if args.s == "oneW":
            data = transform_2_oneWavePsample(data)
        elif args.s == "allW":
            data = transform_2_allWavePsample(data)

    # data = impute_NaN(data, label = label) # impute missing value
    # print(data.isna().mean().sum())

    # ################################### feature selection ###################################
    # if args.f is not None:
    #     if args.f == "ttest":
    #         data, feature_P = select_features_ttest(data, label = label, alpha = 0.001)
    #     elif args.f == "importance":
    #         data, importance = select_features_importance(data, label = label, threshold = 0.01)
    #     elif args.f == "manual":
    #         data = select_features_manual(data, file = "../data/Col_names.txt")
    #     elif args.f == "both":
    #         data = select_features_manual(data, file = "../data/Col_names.txt")
    #         data, feature_P = select_features_ttest(data, label = label, alpha = 0.05)
    #         data, importance = select_features_importance(data, label = label, threshold = 0)
        
    ################################### train test split ###################################
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

    # Output column names for manual selection
    columns_name = data.columns.to_list()
    with open("../output/column_names.txt", 'w') as file:
        for column in columns_name:
            file.write(f"{column} OR\n") # Add OR for library variable search

    # features processing
    print(f"{'*'*30} Select Features {'*'*30}")
    pc_feature_selector = Pipeline([
                        ("correlatedFeatureRemover", correlatedFeatureRemover(upperBound=configs["DataProcessing"]["feature_selection"]["correlationUpperBound"])),
                        ("normalizer", normalizer(lowerBound=configs["DataProcessing"]["feature_selection"]["scaleRange"][0],upperBound=configs["DataProcessing"]["feature_selection"]["scaleRange"][1])), 
                        ("variableFeaturesSelector", variableFeaturesSelector(numFeatures=configs["DataProcessing"]["feature_selection"]["numVariableFeatures"])),
                        ("pca", pcaReducer(criteria=configs["DataProcessing"]["feature_selection"]["pcaCriteria"]))
                        ])

    umap_feature_selector = Pipeline([
                        ("umap", umapReducer(seed=configs["seed"]))
                        ])

    pca_data = pc_feature_selector.fit_transform(data)
    umap_data = umap_feature_selector.fit_transform(pca_data)
    
    #Visualization 
    pca_df = pd.concat([pca_data, target], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=pca_df, x='PC_1', y='PC_2', hue=label)
    plt.title('PCA Scatter Plot')
    plt.savefig("../output/pca_train.png")

    umap_df = pd.concat([umap_data, target], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=umap_df, x='UMAP_1', y='UMAP_2', hue=label)
    plt.title('UMAP Scatter Plot')
    plt.savefig("../output/umap_train.png")

    # -------------------------------------------------------- testing data --------------------------------------------------------
    # data auditing
    data = test_data
    report = audit_data(data, duplicateRow = configs["DataProcessing"]["audit"]["duplicateRow"], duplicateCol = configs["DataProcessing"]["audit"]["duplicateCol"], 
                        NaN = configs["DataProcessing"]["audit"]["NaN"], column_NaN = 0.6,  
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

    # features processing
    pca_data = pc_feature_selector.transform(data)
    umap_data = umap_feature_selector.transform(pca_data)
    
    #Visualization 
    pca_df = pd.concat([pca_data, target], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=pca_df, x='PC_1', y='PC_2', hue=label)
    plt.title('PCA Scatter Plot')
    plt.savefig("../output/pca_test.png")

    umap_df = pd.concat([umap_data, target], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=umap_df, x='UMAP_1', y='UMAP_2', hue=label)
    plt.title('UMAP Scatter Plot')
    plt.savefig("../output/umap_test.png")

if __name__ == "__main__":
    configs = Configs().configs
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-f', type=str, default=None, help='feature selection methods')
    parser.add_argument('-s', type=str, default=None, help='manual selection of features and samples according to specific wave')
    args = parser.parse_args()
    main(configs, args)