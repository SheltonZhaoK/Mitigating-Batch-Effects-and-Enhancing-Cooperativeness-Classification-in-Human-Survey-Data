# -----------------------------------------------------------
# This script provides functionality for auditing, clean, process, and prepare data for downstream analysis
#
# Author: Konghao Zhao, Cade Wiley
# Created: 2023-10-27
# Modified: 2023-10-30
# 
# -----------------------------------------------------------

import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import hstack
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

class pca(BaseEstimator, TransformerMixin):
    def __init__(self, criteria):
        self.criteria = criteria
    
    def fit(self, X, y=None):
        self.pca = PCA(n_components=self.criteria)
        return self
    
    def transform(self, X, y=None):
        print(f"Select {self.criteria} principle component")
        pca_data = self.pca.fit_transform(X)
        print(f"Dataset shape: {pca_data.shape}")
        pca_data = pd.DataFrame(pca_data)
        pca_data.index = X.index.to_list()
        pca_data.columns = [f"PC_{i+1}" for i in range(pca_data.shape[1])]
        pca_data.index.name = X.index.name
        return X

class variableFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, numFeatures):
        self.numFeatures = numFeatures
    
    def fit(self, X, y=None):
        columns_std = X.std()
        sorted_columns = columns_std.sort_values(ascending=False).index
        self.highly_variable_columns = sorted_columns[:self.numFeatures]
        return self
    
    def transform(self, X, y=None):
        print(f"Select {self.numFeatures} highly variable features")
        X = X[self.highly_variable_columns]
        print(f"Dataset shape: {X.shape}")
        return X
    
class normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler(feature_range=(self.lowerBound, self.upperBound))
        return self
    
    def transform(self, X, y=None):
        print(f"Normalize Dataset")
        np_data = self.scaler.fit_transform(X)
        X = X.copy()
        X.iloc[:,:] = np_data
        return X

class correlatedFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, upperBound):
        self.upperBound = upperBound
    
    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        self.columns_to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.upperBound:
                    colname = corr_matrix.columns[i]
                    self.columns_to_drop.add(colname)
        return self
    
    def transform(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop)
        print(f"Drop {len(self.columns_to_drop)} highly correlated columns with {self.upperBound} upper_bound")
        print(f"Dataset shape: {X.shape}")
        return X


class rowOutlierChecker(BaseEstimator, TransformerMixin):
    def __init__(self, stdScale, evaluation):
        self.stdScale = stdScale
        self.evaluation = evaluation
        
    
    def fit(self, X, y=None):
        row_mean = X.mean(axis=1)
        mean = row_mean.mean()
        std = row_mean.std()

        self.upper_bound = mean + self.stdScale*std
        self.lower_bound = mean - self.stdScale*std
        return self

    def transform(self, X, y=None):
        print(f"{'-'*10} Filter Rows {'-'*10}")
        X['mean'] = X.mean(axis=1)
        rows_to_remove = X[(X['mean'] < (self.lower_bound)) | (X['mean'] > (self.upper_bound))].index
        if self.evaluation:
            print(f"Find {len(rows_to_remove)} out of distribution rows in evaluation, not dropped")
        else:
            X = X.drop(rows_to_remove)
            print(f"Drop {len(rows_to_remove)} out of distribution rows")
        X = X.drop(columns=["mean"])
        return X
    
class NaColumnsHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columnNames):
        self.columnNames = columnNames
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"{'-'*10} Handle Large NaN columns {'-'*10}")
        if self.columnNames is not None:
            X = X.drop(columns = self.columnNames)
            print(f"Drop {len(self.columnNames)} columns that have execessive N/As")
            print(f"Dataset shape: {X.shape}")
        return X

class NaHandler(BaseEstimator, TransformerMixin):
    def __init__(self, indices, evaluation=False):
        self.row_indices = indices[0]
        self.column_indices = indices[1]
        self.evaluation = evaluation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"{'-'*10} Handle NaN values {'-'*10}")
        if not len(self.row_indices) == 0:
            X = X.drop(self.row_indices)
            if self.evaluation:
                print(f"{len(self.row_indices)} samples have N/A that can't be predicted are removed")
        print(f"Drop {len(self.row_indices)} rows with N/As")
        print(f"Dataset shape: {X.shape}")
        return X

class duplicateHandler(BaseEstimator, TransformerMixin):
    def __init__(self, indices, evaluation=False):
        self.row_indices = indices[0]
        self.column_indices = indices[1]
        self.evaluation = evaluation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"{'-'*10} Handle duplicates {'-'*10}")
        if not len(self.row_indices) == 0:
            if self.evaluation:
                print(f"Find {len(self.row_indices)} duplicated samples, not removed in evaluation")
            else:
                X = X.drop(self.row_indices)

        if not len(self.column_indices) == 0:
            if self.evaluation:
                print(f"Find {len(self.column_indices)} duplicated samples, not removed in evaluation")
            else:
                X = X.drop(columns=self.column_indices)
        print(f"Dataset shape: {X.shape}")
        return X

        
class dataBalancer(BaseEstimator, TransformerMixin):
    def __init__(self, metaData, column2balance, outputDir, training=True, ):
        self.metaData = metaData.copy().reset_index(drop=True)
        self.column2balance = column2balance
        self.training = training
        self.outputDir = outputDir

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.training:
            print(f"{'='*20} Balance rows for training data{'='*20}")
            sample_size = self.metaData[self.column2balance].value_counts().min()
            ix = []
            for x in self.metaData[self.column2balance].unique():
                indices = self.metaData[self.metaData[self.column2balance] == x].index
                ix.extend(np.random.choice(indices, sample_size, replace=False))
            np.save(os.path.join(self.outputDir, "training_indices.npy"), ix)
            # X = X.tocsr()[ix].tocsc()
            X = X[ix]
            return X 
        else:
            print(f"{'='*20} Rows not balanced for test data {'='*20}")
            return X

class sparseNullRemover(TransformerMixin, BaseEstimator):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        remove = []
        num_cols = X.shape[1]
        num_rows = X.shape[0]
        for col in range(num_cols):
            zero = num_rows - (X.getcol(col)).count_nonzero()
            if zero > (num_rows * self.cutoff):
                remove.append(col)
        self.columns_to_keep = [col for col in range(num_cols) if col not in remove]
        return self
    
    def transform(self, X, y=None):
        print(f"{'='*20} Filter out columns that have large NAs in sparse{'='*20}")
        new_data = hstack([X.getcol(col) for col in self.columns_to_keep], format='csc')
        return new_data

class PdConverter(TransformerMixin, BaseEstimator):
    def __init__(self, index, inputDir=None):
        self.index = index
        self.inputDir = inputDir

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.inputDir is not None:
            training_indices = np.load(os.path.join(self.inputDir, 'training_indices.npy'))
            self.index = [self.index[i] for i in training_indices]
    
        data = pd.DataFrame(X)
        data.index = self.index
        data.columns = [f"PC_{i+1}" for i in range(X.shape[1])]
        return data

class StandardizeScaler(TransformerMixin, BaseEstimator):
    def __init__(self, mean):
        self.mean = mean

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"{'='*20} Standardize Data {'='*20}")
        stdScaler = StandardScaler(with_mean=self.mean)
        X = stdScaler.fit_transform(X)
        return X
    
class logNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(f"{'='*20} Log Normalize Data {'='*20}")
        return np.log1p(X)

class rowFilter(TransformerMixin, BaseEstimator):
    def __init__(self, std_scale = 3, outOfDist = True, evaluation = False):
        self.std_scale = std_scale
        self.outOfDist = outOfDist
        self.evaluation = evaluation

    def fit(self, X, y=None):
        if self.outOfDist is not None:
            mean = X.mean(axis=1).mean()
            std = X.mean(axis=1).std()
            self.upper_bound = mean + self.std_scale*std
            self.lower_bound = mean - self.std_scale*std
        return self

    def transform(self, X, y=None):
        print(f"{'='*20} Filter Rows {'='*20}")
        
        X['mean'] = X.mean(axis=1)
        if not self.evaluation:
            if self.outOfDist:
                rows_to_remove = X[(X['mean'] < (self.lower_bound)) | (X['mean'] > (self.upper_bound))].index
                X = X.drop(rows_to_remove)
                print(f"Drop {len(rows_to_remove)} out of distribution rows")
            X = X.drop(columns=["mean"])
            print(X.shape)
            return X
        else:
            if self.outOfDist:
                rows_to_remove = X[(X['mean'] < (self.lower_bound)) | (X['mean'] > (self.upper_bound))].index
                print(f"Find {len(rows_to_remove)} out of distribution rows in evaluation, not dropped")
            X = X.drop(columns=["mean"])
            print(X.shape)
            return X

def check_column_NA(data, cutoff):
    na_ratios = data.isna().mean()
    columns2drop = na_ratios[na_ratios > cutoff].index.tolist()
    if len(columns2drop) > 0:
        return columns2drop
    else:
        return None


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