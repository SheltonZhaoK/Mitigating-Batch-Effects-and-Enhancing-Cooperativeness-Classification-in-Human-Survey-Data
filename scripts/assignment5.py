import warnings, os
import pandas as pd
import urllib.request as ur

from sklearn.pipeline import Pipeline
from dataObject import NaColumnsHandler, NaHandler, duplicateHandler, rowOutlierChecker, correlatedFeatureRemover, normalizer, variableFeaturesSelector, pca, audit_data
# from data import audit_data

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

#################################### features processing ###################################
print(f"{'*'*30} Select Features {'*'*30}")
feature_selector = Pipeline([
                    ("correlatedFeatureRemover", correlatedFeatureRemover(upperBound=0.98)),
                    ("normalizer", normalizer(lowerBound=-1,upperBound=1)), 
                    ("variableFeaturesSelector", variableFeaturesSelector(numFeatures=100)),
                    ("pca", pca(criteria=0.98))
                    ])
data = feature_selector.fit_transform(data)

# columns_name = data.columns.to_list()
# with open("../output/column_names.txt", 'w') as file:
#     for column in columns_name:
#         file.write(f"{column} OR\n")
