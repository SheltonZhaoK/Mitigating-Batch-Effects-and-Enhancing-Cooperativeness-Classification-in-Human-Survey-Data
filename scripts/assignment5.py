import warnings, os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request as ur

from sklearn.pipeline import Pipeline
from dataObject import (NaColumnsHandler, NaHandler, duplicateHandler, rowOutlierChecker, correlatedFeatureRemover, 
                        normalizer, variableFeaturesSelector, pcaReducer, audit_data, umapReducer)

def main():
    ################################### read data ###################################
    label = "W2CL1R"
    data = pd.read_csv("../data/ICPSR_36371/DS0001/36371-0001-Data.tsv", sep="\t", na_values=[" "])

    #################################### data auditing ###################################
    report = audit_data(data, duplicateRow = True, duplicateCol = True, NaN = True, column_NaN = True,  maxNA = 0.51)

    #################################### data cleaning ###################################
    print(f"{'*'*30} Data Cleaning {'*'*30}")
    dataCleaner = Pipeline([
                            ("NaColumnsHandler", NaColumnsHandler(columnNames=report["Large_NAs_columns"])),
                            ("NaHandler", NaHandler(indices=report["NaN"], evaluation=False)),
                            ("logNormalizer", duplicateHandler(indices=report["duplicate"], evaluation=False)),
                            ("rowOutlierChecker", rowOutlierChecker(stdScale=3, evaluation=False)),
                            ])
    data = dataCleaner.fit_transform(data)

    target = data[[label]]
    data = data.drop(columns=[label])

    # Output column names for manual selection
    columns_name = data.columns.to_list()
    with open("../output/column_names.txt", 'w') as file:
        for column in columns_name:
            file.write(f"{column} OR\n") # Add OR for library variable search

    #################################### features processing ###################################
    print(f"{'*'*30} Select Features {'*'*30}")
    pc_feature_selector = Pipeline([
                        ("correlatedFeatureRemover", correlatedFeatureRemover(upperBound=0.98)),
                        ("normalizer", normalizer(lowerBound=-1,upperBound=1)), 
                        ("variableFeaturesSelector", variableFeaturesSelector(numFeatures=100)),
                        ("pca", pcaReducer(criteria=0.98))
                        ])

    umap_feature_selector = Pipeline([
                        ("umap", umapReducer(seed=1))
                        ])

    pca_data = pc_feature_selector.fit_transform(data)
    umap_data = umap_feature_selector.fit_transform(pca_data)
    
    #################################### Visualization ###################################
    pca_df = pd.concat([pca_data, target.reset_index(drop=True)], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=pca_df, x='PC_1', y='PC_2', hue=label)
    plt.title('PCA Scatter Plot')
    plt.show()

    umap_df = pd.concat([umap_data, target.reset_index(drop=True)], axis=1)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=umap_df, x='UMAP_1', y='UMAP_2', hue=label)
    plt.title('UMAP Scatter Plot')
    plt.show()

if __name__ == "__main__":
    main()