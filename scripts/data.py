# -----------------------------------------------------------
# This script provides functionality for auditing, clean, process, and prepare data for downstream analysis
#
# Author: Konghao Zhao
# Created: 2023-09-20
# Modified: 2023-10-4
# 
# -----------------------------------------------------------

import sys, os, ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from multiprocessing import Pool

def one_hot_encoding(data, columns = None, outputDir = None, inputDir = None):
    print(f"{'*'*30} One-hot encoding {'*'*30}")
    data = pd.get_dummies(data, columns = columns)
    if outputDir is not None:
        data.to_csv(os.path.join(outputDir, "one_hot_feature.csv"), index = False)
        print(f"{'<'*10} One-hot encoding features saved to {os.path.join(outputDir, 'one_hot_feature.csv')} {'>'*10}")
    if inputDir is not None:
        features = pd.read_csv(os.path.join(inputDir, "one_hot_feature.csv"))
        # assert len(features.columns.to_list()) == len(data.columns.to_list()), "Can't process evaluation data, features after one-hot encoding do not match"
        assert set(features.columns).issubset(set(data.columns)), "Can't process evaluation data, features does not match"
        data = data[features.columns.to_list()]
        print(f"{'<'*10} Match features to {os.path.join(inputDir, 'one_hot_feature.csv')} {'>'*10}")
    return data

def split_features(data, cutoff = 5):
    # print(data.nunique())
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    # rest = data.drop(columns=categorical_features)
    # categorical_features.extend(rest.columns[rest.nunique() <= cutoff].tolist())
    numeric_features = list(set(data.columns.to_list()) - set(categorical_features))
    print(f"Numeric features: {numeric_features}\nCategorical features: {categorical_features}")
    return data[numeric_features], data[categorical_features]

def check_column_NA(data, cutoff):
    na_ratios = data.isna().mean()
    columns2drop = na_ratios[na_ratios > cutoff].index.tolist()
    if len(columns2drop) > 0:
        return columns2drop
    else:
        return None

def sanity_check(data, maxNA = 0.5):
    columns2drop = check_column_NA(data, maxNA)
    if columns2drop is not None:
        data = data.drop(columns2drop, axis=1)
        print(f"Sanity check ——> {columns2drop} with more than {maxNA*100}% N/A are dropped")
    return data

'''
Detect the encoding of a file by try and errors, exits if the encoding can't be found
    :param filePath: a string of path to the file
    :param delimiter: a character of delimiter in the file
    
    :returns encoding: a string of the encoding of the file
'''
def detect_encoding(filePath, delimiter):
    for encoding in ['utf-8','utf-16','utf-32','latin-1','ascii','cp1252','iso-8859-1',
                     'iso-8859-2','iso-8859-15','cp437','cp850','mac_roman','gb2312','gbk','big5',
                     'shift_jis','euc_jp','euc_kr']:
        try:
            data = pd.read_csv(filePath, encoding=encoding, delimiter = delimiter)
            return encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    print("File encoding can not be decided")
    exit()

'''
Read data from file
    :param filePath: a string of path to the file
    :param index: a string of the column name as index 
    :param label: a string of the column name as label 
    :param dropped_columns: a list of the column name to drop
    :param delimiter: a character of delimiter in the file 
    
    :returns data: a panda dataFrame of data
    :returns labels: a panda dataFrame of labels
'''
def read_data(filePath, label, index = None, dropped_columns = None, encoding = False, delimiter = ",", expand = None):
    if encoding:
        encoding = detect_encoding(filePath, delimiter)
        print(f"Read {filePath} with {encoding} encoding")
        if index is None:
            data = pd.read_csv(filePath, index_col = "Unnamed: 0", encoding = encoding, delimiter = delimiter)      
        else:
            data = pd.read_csv(filePath, index_col = index, encoding = encoding, delimiter = delimiter)
    else:
        print(f"Read {filePath} ")
        if index is None:
            data = pd.read_csv(filePath, index_col = "Unnamed: 0", delimiter = delimiter)
        else:
            data = pd.read_csv(filePath, index_col = index, delimiter = delimiter)

    if expand is not None:
        expanded_columns = data[expand].apply(ast.literal_eval).apply(pd.Series)
        data = pd.concat([data.drop(expand, axis=1), expanded_columns], axis=1)   

    data.index.name = "ID"
    labels = None
    if label is not None:
        data = data.dropna(subset=[label])
        labels = data[[label]]
        labels.index = data.index.to_list()
        labels.columns = [label]
        data = data.drop(columns=label)
        labels = labels.rename(columns={label: 'labels'})
    if dropped_columns is not None:
        data = data.drop(columns=dropped_columns)
    return data, labels

'''
Find NA values in a file
    :param data: a panda dataFrame
    
    :returns [row_indices, None]: a list only containing the index of NAs
'''
def find_NaN_value(data):
    if data.isna().any().any():
        row_indices, col_indices = np.where(data.isna()) 
        # row_indices = list(set(data.index[row_indices]))
        row_indices = list(data.index[row_indices])
        col_names = data.columns[col_indices].to_list()
        return [row_indices, col_names]
    else:
        return None


'''
Find duplicated rows or columns in a file
    :param data: a panda dataFrame
    :param duplicateRow: a string of the column name as index 
    :param duplicateCol: a string of the column name as label 
    
    :returns [duplicate_rows, duplicate_columns]: a list of row and column indices of duplicated columns
'''
def find_duplicate(data, duplicateRow = True, duplicateCol = True):
    if duplicateRow:
        duplicate_rows = data[data.duplicated(keep=False)].index.to_list()
    else:
        duplicate_rows = []
        
    if duplicateCol:
        duplicate_columns = data.T[data.T.duplicated(keep=False)].index.to_list()
    else:
        duplicate_columns = []
    
    if not len(duplicate_rows) == 0 or not len(duplicate_columns) == 0:
        return [duplicate_rows, duplicate_columns]
    else:
        return None

'''
Check if the data's column and index match with the given one
    :param data: a panda dataFrame
    :param columns_name: a list of columns_name
    :param index_name: a list of index
    
    :returns boolean: a boolean if the name matches
'''
def check_alignment(data, columns_name, index_name):
    if data.columns.to_list() == columns_name and data.index.to_list() == index_name:
        return True
    else:
        return False

'''
Audit the data quality issue for data cleaning and generate a report
    :param data: a panda dataFrame
    :param duplicateRow: a boolean to decide whether to check for duplicated row
    :param duplicateCol: a boolean to decide whether to check for duplicated columns
    :param duplicateCol: a boolean to decide whether to check for NaN
    
    :returns report: a dictionary containing the issue as key and indices as value
'''
def audit_data(data, duplicateRow = True, duplicateCol = True, NaN = True, column_NaN = True, maxNA = 0.5):
    print(f"{'*'*30} Data Auditing {'*'*30}")
    columns_name, index_name = None, None
    report = {}
    
    if column_NaN:
        columns2drop = check_column_NA(data, maxNA)
        if columns2drop is not None:
            report["Large_NAs_columns"] = columns2drop
            print(f"Find {len(columns2drop)} columns with more than {maxNA*100}% N/A")

    if NaN:
        NaN_indices = find_NaN_value(data)
        if NaN_indices is not None:
            if "Large_NAs_columns" in report:
                filtered_rows = []
                filtered_columns = []
                for row_idx, col_name in zip(NaN_indices[0], NaN_indices[1]):
                    if col_name not in columns2drop:
                        filtered_rows.append(row_idx)
                        filtered_columns.append(col_name)
            report["NaN"] = [list(set(filtered_rows)), list(set(filtered_columns))]
    
    if duplicateRow or duplicateCol:
        duplicate_indices = find_duplicate(data, duplicateRow, duplicateCol)
        if duplicate_indices is not None:
            report["duplicate"] = duplicate_indices
    
    for key in report:
        if key == "NaN":
            print(f"Find {len(set(NaN_indices[0]))} rows with NaN")
        if key == "duplicate":
            if duplicate_indices is not None:
                if not len(duplicate_indices[0]) == 0:
                    print(f"Find {len(duplicate_indices[0])} duplicated rows")
                if not len(duplicate_indices[1]) == 0:
                    print(f"Find {len(duplicate_indices[1])} duplicated columns")
    if len(report) == 0:
        print(f"No N/A or duplicates are found")
    
    if columns2drop is not None:
        report["duplicate"][1] = [index for index in report["duplicate"][1] if index not in report["Large_NAs_columns"]]
    return report

'''
Clean the data given the auditing report
    :param data: a panda dataFrame
    :param label: a string of label name
    :param labels: a panda dataFrame
    :param report: a dictionary from audit_data()
    :param classification: a boolean to decide whether to enode the labels as numeric numebrs
    
    :returns data: a panda dataFrame
    :param labels: a panda dataFrame
'''
def clean_data(data, labels, report, classification = True, evaluation = False):
    print(f"{'*'*30} Data Cleaning {'*'*30}")
    num_rows = 0
    num_cols = 0
    label = "labels"
    if labels is not None:
        if classification:
            labels = labels.iloc[:,0].astype('category').cat.codes.to_frame(label)
    print(f"Original dataset shape: {data.shape}")

    for key in report:
        print(f"{'-'*10} Handle {key}{'-'*10}")

        if key == "Large_NAs_columns":
            data = data.drop(columns = report["Large_NAs_columns"])
            num_cols += len(report["Large_NAs_columns"])
            print(f"Drop {len(report['Large_NAs_columns'])} columns that have execessive N/As")
            print(f"Dataset shape: {data.shape}")
            continue

        row_indices = report[key][0]
        column_indices = report[key][1]
        if key == "NaN":
            if not len(row_indices) == 0:
                data = data.drop(row_indices)
                if labels is not None:
                    labels = labels.drop(row_indices)
                    if evaluation:
                        print(f"{len(row_indices)} samples have N/A that can't be predicted are removed")
                num_rows += len(row_indices)
        elif key == "duplicate":
            if not len(row_indices) == 0:
                if evaluation:
                    print(f"Find {len(row_indices)} duplicated samples, not removed in evaluation")
                data = data.drop(row_indices)
                if labels is not None:
                    labels = labels.drop(row_indices)
                num_rows += len(row_indices)
            if not len(column_indices) == 0:
                data = data.drop(columns=column_indices)
                num_cols += len(column_indices)
        print(f"Dataset shape: {data.shape}")
    print(f"Remove {num_rows} rows {num_cols} columns in total")
    if labels is not None:
        assert data.index.to_list() == labels.index.to_list(), "data and label indices are not matched" 
    return data, labels

'''
Drop rows that hev mean outside the range
    :param data: a panda dataFrame
    :param labels: a panda dataFrame
    :param std_scale: a integer that specify the range eg: [-std_scale * mean, std_scale * mean]
    :param outputDir: a string of output directory of ranges if it is not None
    :param inputDir: a string of input directory of ranges if it is not None
    
    :returns data: a panda dataFrame
    :param labels: a panda dataFrame
'''
def drop_outOfDistribution_rows(data, labels, std_scale, outputDir = None, inputDir = None, evaluation = False):
    if inputDir is not None:
        data['mean'] = data.mean(axis=1)
        mean = data['mean'].mean()
        ranges = np.loadtxt(os.path.join(inputDir, "mean_range.txt"))
        print(f"Row mean ranges extracted from {os.path.join(inputDir, 'mean_range.txt')}")
        lower_bound = ranges[0]
        upper_bound = ranges[1]
    else:
        data['mean'] = data.mean(axis=1)
        mean = data['mean'].mean()
        std = data['mean'].std()
        upper_bound = mean + std_scale*std
        lower_bound = mean - std_scale*std
    rows_to_remove = data[(data['mean'] < (lower_bound)) | (data['mean'] > (upper_bound))].index
    if evaluation:
        print(f"Find {len(rows_to_remove)} out of distribution rows in evaluation, not dropped")
    else:
        data = data.drop(rows_to_remove)
        if labels is not None:
            labels = labels.drop(rows_to_remove)
        print(f"Drop {len(rows_to_remove)} out of distribution rows")
    data = data.drop(columns=["mean"])
    if labels is not None:
        assert data.index.to_list() == labels.index.to_list(), "data and label indices are not matched"
    if outputDir is not None:
        mean_range = np.array([mean - std_scale*std, mean + std_scale*std])
        np.savetxt(os.path.join(outputDir, "mean_range.txt"), mean_range, delimiter=",")
        print(f"{'<'*10} Row mean range saved to {os.path.join(outputDir, 'mean_range.txt')} {'>'*10}")
    return data, labels

def drop_rows_w_outlier(data, labels, z_range):
    row_means = data.mean(axis=1)
    row_stds = data.std(axis=1)
    z_scores = data.subtract(row_means, axis=0).divide(row_stds, axis=0)
    rows_with_outliers = z_scores[(z_scores > z_range) | (z_scores < -z_range)].any(axis=1)
    data = data[~rows_with_outliers]
    if labels is not None:
        labels = labels[~rows_with_outliers]
    print(f"Drop {rows_with_outliers.sum()} rows that have outliers")
    if labels is not None:
        assert data.index.to_list() == labels.index.to_list(), "data and label indices are not matched" 
    return data, labels

'''
Wrapper funciton for filtering rows
    :param data: a panda dataFrame
    :param labels: a panda dataFrame
    :param std_scale: a integer that specify the range eg: [-std_scale * mean, std_scale * mean]
    :param outOfDist: a boolean that specify whether to drop out of distribution rows
    :param outlier: a boolean that specify whether to drop rows that have outliers
    :param visualize: a boolean that specify whether to visualize the process of filtering
    :param outputDir: a string of output directory of log if it is not None
    :param inputDir: a string of input directory of log if it is not None
    
    :returns data: a panda dataFrame
    :param labels: a panda dataFrame
'''
def filter_rows(data, labels, std_scale = 3, z_range = 3, outOfDist = True, outlier = False, visualize = False, outputDir = None, inputDir = None, evaluation = False):
    print(f"{'*'*30} Data Processing {'*'*30}")
    print(f"{'-'*10} Filter Rows {'-'*10}")
    if visualize:
        print(f"Original Distribution")
        data.T.boxplot()
        plt.show()
    
    # drop rows that are out of the distribution based on the z-values of each row
    if outOfDist:
        data, labels = drop_outOfDistribution_rows(data, labels, std_scale, outputDir, inputDir, evaluation)
        print(f"Dataset shape: {data.shape}")
        if visualize:
            data.T.boxplot()
            plt.show()
    
    # drop rows that have outliers based on the z-values of each element in a row
    if outlier:
        data, labels = drop_rows_w_outlier(data, labels, z_range)
        print(f"Dataset shape: {data.shape}")
        if visualize:
            data.T.boxplot()
            plt.show()
    return data, labels

'''
Remove columns that have zero larger than the threshold
    :param data: a panda dataFrame
    :param zero_percentage: a float of percentage of zeros

    :param labels: a panda dataFrame
'''
def remove_columns_w_zeros(data, zero_percentage = 0.5):
    cols_to_drop = [col for col in data.columns if (data[col] == 0).sum() > (zero_percentage * len(data))]
    data = data.drop(columns=cols_to_drop)
    print(f"Drop {len(cols_to_drop)} columns that have more than {zero_percentage*100}% zeros")
    return data

'''
Normalize data
    :param data: a panda dataFrame
    :param rangeV: a list of floats specifying the scaling range
    :param outputDir: a string of output directory of scaler object if it is not None
    :param inputDir: a string of input directory of scaler object if it is not None
    
    :returns data: a panda dataFrame
'''
def normalize_data(data, rangeV, outputDir = None, inputDir = None):
    if inputDir is not None:
        scaler = load(os.path.join(inputDir, "normalizer.joblib"))
        print(f"Normalize Dataset using scaler from {os.path.join(inputDir, 'normalizer.joblib')}")
    else:
        print(f"Normalize Dataset")
        minV = rangeV[0]
        maxV = rangeV[1]
        scaler = MinMaxScaler(feature_range=(minV, maxV))
    np_data = scaler.fit_transform(data)
    data = data.copy()
    data.iloc[:,:] = np_data
    if outputDir is not None:
        dump(scaler, os.path.join(outputDir, "normalizer.joblib"))
        print(f"{'<'*10} Row mean range saved to {os.path.join(outputDir, 'mean_range.txt')} {'>'*10}")
    return data

'''
Select highly variable features based on the std computed for each column
    :param data: a panda dataFrame
    :param number: a number of feature with highest std selected
    :param outputDir: a string of output directory of processed data if it is not None
    :param inputDir: a string of input directory of processed data if it is not None
    
    :returns data: a panda dataFrame
'''
def select_highly_variable_features(data, number = 50, outputDir = None, inputDir = None):
    if inputDir is not None:
        mapping = pd.read_csv(os.path.join(inputDir, "select_highly_variable_features.csv"))
        columns_mapping = mapping.columns.to_list()
        data = data[columns_mapping]
        print(f"Select {len(columns_mapping)} highly variable columns from {os.path.join(inputDir, 'select_highly_variable_features.csv')}")
    else:
        columns_std = data.std()
        sorted_columns = columns_std.sort_values(ascending=False).index
        highly_variable_columns = sorted_columns[:number]
        data = data[highly_variable_columns]
        if number > len(data.columns):
            print(f"Can't find {number} variable features, keep the original features")
        else:
            print(f"Select {number} highly variable features")
        if outputDir is not None:
            data.to_csv(os.path.join(outputDir, "select_highly_variable_features.csv"), index = False)
            print(f"{'<'*10} Data after selected highly variable features saved to {os.path.join(outputDir, 'remove_correlated_features.csv')} {'>'*10}")
    return data

'''
Run pca and automatically select number of PCs
    :param data: a panda dataFrame
    :param num_components: a number of initial principal components to retain
    :param cumulative_variance_ratio: a float of cumulative variance PCs should reach
    :param visualize: a boolean specify whether to visualize the data
    
    :returns data: a panda dataFrame
'''
def run_pca(data, num_components = 50, cumulative_variance_ratio = 0.98, visualize = False):
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(data)
    explained_ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_pcs_retained = np.argmax(cumulative_ratio >= cumulative_variance_ratio)
    if n_pcs_retained == 0:
        n_pcs_retained = num_components
        
    if n_pcs_retained < int(len(data) / 5):
        pca_data = pca_data[:, :n_pcs_retained]
        print(f"Select {n_pcs_retained} principal components with {cumulative_variance_ratio} cumulative variance")
        print(f"Dataset shape: {pca_data.shape}")
    else:
        pca_data = pca_data[:, :int(len(data) / 5)]
        print(f"Select {int(len(data)/5)} principal components {cumulative_variance_ratio} cumulative variance")
        print(f"Dataset shape: {pca_data.shape}")
    
    pca_data = pd.DataFrame(pca_data)
    pca_data.index = data.index.to_list()
    pca_data.columns = [f"PC_{i+1}" for i in range(pca_data.shape[1])]
    pca_data.index.name = data.index.name
    
    # If visualization is desired
    if visualize:
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio, marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=cumulative_variance_ratio, color='r', linestyle='--')
        plt.show()
    return pca_data


'''
Wrapper funciton for selecting features
    :param data: a panda dataFrame
    :param zero_percentage: a float of percentage of zeros
    :param correlation_upper_bound: a float of maximum correlation between columns
    :param highly_variable_features: a number of feature with highest std selected
    :param pca: a boolean that specify whether perform pca
    :param num_components: a number of initial principal components to retain
    :param cumulative_variance_ratio: a float of cumulative variance PCs should reach
    :param normalize: a list of floats specifying the scaling range if not None
    :param visualize: a boolean that specify whether to visualize the process of selection
    :param outputDir: a string of output directory of log if it is not None
    :param inputDir: a string of input directory of log if it is not None
    :param drop_zeros: a boolean that specify whether to drop_zeros
    
    :returns data: a panda dataFrame
'''
def select_features(data, zero_percentage = 0.5, correlation_upper_bound = 0.98, highly_variable_features = 100, 
                    pca = False, num_components = 30, cumulative_variance_ratio = 0.98, normalize = None, visualize = False,
                    outputDir = None, inputDir = None, drop_zeros = True):
    print(f"{'-'*10} Select Columns {'-'*10}")
    if drop_zeros:
        data = remove_columns_w_zeros(data, zero_percentage = zero_percentage)
        print(f"Dataset shape: {data.shape}")
    data = remove_correlated_features(data, upper_bound = correlation_upper_bound, outputDir = outputDir, inputDir = inputDir)
    print(f"Dataset shape: {data.shape}")
    if normalize is not None:
        data = normalize_data(data, rangeV = normalize, outputDir = outputDir, inputDir = inputDir)
    data = select_highly_variable_features(data, number = highly_variable_features, outputDir = outputDir, inputDir = inputDir)
    print(f"Dataset shape: {data.shape}")
    if pca:
        data = run_pca(data, num_components = num_components, cumulative_variance_ratio = cumulative_variance_ratio,
                       visualize = visualize)
    return data

'''
Remove highly correlated features
    :param data: a panda dataFrame
    :param upper_bound: a float of maximum correlation between columns
    :param outputDir: a string of output directory of processed data if it is not None
    :param inputDir: a string of input directory of processed data if it is not None
    
    :returns data: a panda dataFrame
'''
def remove_correlated_features(data, upper_bound = 0.99, outputDir = None, inputDir = None):
    if inputDir is not None:
        mapping = pd.read_csv(os.path.join(inputDir, "remove_correlated_features.csv"))
        columns_mapping = mapping.columns.to_list()
        data = data[columns_mapping]
        print(f"Select {len(columns_mapping)} none correlated columns from {os.path.join(inputDir, 'remove_correlated_features.csv')}")
    else:
        corr_matrix = data.corr().abs()
        columns_to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > upper_bound:
                    colname = corr_matrix.columns[i]
                    columns_to_drop.add(colname)
        print(f"Drop {len(columns_to_drop)} highly correlated columns with {upper_bound} upper_bound")
        data = data.drop(columns=columns_to_drop)
        if outputDir is not None:
            data.to_csv(os.path.join(outputDir, "remove_correlated_features.csv"), index = False)
            print(f"{'<'*10} Data after removing correlated features saved to {os.path.join(outputDir, 'remove_correlated_features.csv')} {'>'*10}")
    return data

'''
Balance dataset
    :param data: a panda dataFrame
    :param labels: a panda dataFrame

    :return data: a panda dataFrame
    :return labels: a panda dataFrame
'''
def balance_dataset(data, labels):
    print(f"{'-'*10} Balance Dataset {'-'*10}")
    label = labels.columns[0]
    min_labels = labels[label].value_counts().min()
    ix = []
    for x in pd.unique(labels[label]):
        ix.extend(labels[labels[label] == x].sample(n=min_labels).index)
    data = data.loc[ix]
    labels = labels.loc[ix]
    print(f"Dataset shape: {data.shape}")
    return data, labels