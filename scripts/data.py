# -----------------------------------------------------------
# This script provides functionality for auditing, clean, process, and prepare data for downstream analysis
#
# Author: Konghao Zhao, Cade Wiley
# Created: 2023-10-27
# Modified: 2023-10-30
# 
# -----------------------------------------------------------

import os, umap, json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

class umapReducer(BaseEstimator, TransformerMixin):
    def __init__(self, seed):
        self.seed = seed
    
    def fit(self, X, y=None):
        self.umap = umap.UMAP(random_state=self.seed).fit(X)
        return self
    
    def transform(self, X, y=None):
        print(f"Reduce data to 2 UMAP component")
        umap_data = self.umap.transform(X)
        umap_data = pd.DataFrame(umap_data)
        umap_data.index = X.index.to_list()
        umap_data.columns = ["UMAP_1", "UMAP_2"]
        return umap_data
    
class pcaReducer(BaseEstimator, TransformerMixin):
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
        return pca_data

class variableFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, numFeatures):
        self.numFeatures = numFeatures
    
    def fit(self, X, y=None):
        columns_std = X.std()
        sorted_columns = columns_std.sort_values(ascending=False).index
        self.highly_variable_columns = sorted_columns[:self.numFeatures]
        return self
    
    def transform(self, X, y=None):
        if len(self.highly_variable_columns) < self.numFeatures:
            print(f"Can't Select {self.numFeatures} highly variable features, keep the original features")
        else:
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
        # self.columns_to_keep = [column for column in X.columns if column not in self.columns_to_drop]
        return self
    
    def transform(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop)
        # X = X[[self.columns_to_keep]]
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
        print(f"Dataset shape: {X.shape}")
        return X
    
class NaColumnsHandler(BaseEstimator, TransformerMixin):
    def __init__(self, report):
        self.report = report
       
    def fit(self, X, y=None):
        if "Large_NAs_columns" in self.report:
            self.columnNames = self.report["Large_NAs_columns"]
            self.proceed = True
        else:
            self.proceed = False
        return self
    
    def transform(self, X, y=None):
        if self.proceed:
            print(f"{'-'*10} Handle Large NaN columns {'-'*10}")
            if self.columnNames is not None:
                X = X.drop(columns = self.columnNames)
                print(f"Drop {len(self.columnNames)} columns that have execessive N/As")
                print(f"Dataset shape: {X.shape}")
        return X

class NaHandler(BaseEstimator, TransformerMixin):
    def __init__(self, report, evaluation=False):
        self.report = report
        self.evaluation = evaluation
    
    def fit(self, X, y=None):
        if "NaN" in self.report:
            indices = self.report["NaN"]
            self.row_indices = indices[0]
            self.column_indices = indices[1]
            self.proceed = True
        else:
            self.proceed = False
    
        return self
    
    def transform(self, X, y=None):
        if self.proceed:
            print(f"{'-'*10} Handle NaN values {'-'*10}")
            if not len(self.row_indices) == 0:
                X = X.drop(self.row_indices)
                if self.evaluation:
                    print(f"{len(self.row_indices)} samples have N/A that can't be predicted are removed")
            print(f"Drop {len(self.row_indices)} rows with N/As")
            print(f"Dataset shape: {X.shape}")
        return X

class duplicateHandler(BaseEstimator, TransformerMixin):
    def __init__(self, report, evaluation=False):
        self.report = report
        self.evaluation = evaluation
    
    def fit(self, X, y=None):
        if "duplicate" in self.report:
            indices = self.report["duplicate"]
            self.row_indices = indices[0]
            self.column_indices = indices[1]
            self.proceed = True
        else:
            self.proceed = False
        return self
    
    def transform(self, X, y=None):
        if self.proceed:
            print(f"{'-'*10} Handle duplicates {'-'*10}")
            if not len(self.row_indices) == 0:
                if self.evaluation:
                    print(f"Find {len(self.row_indices)} duplicated samples, not removed in evaluation")
                else:
                    X = X.drop(self.row_indices)

            if not len(self.column_indices) == 0:
                if self.evaluation:
                    print(f"Find {len(self.column_indices)} duplicated columns, not removed in evaluation")
                else:
                    X = X.drop(columns=self.column_indices)
            print(f"Dataset shape: {X.shape}")
        return X

        
class dataBalancer(BaseEstimator, TransformerMixin):
    def __init__(self, metaData, column2balance, outputDir, training=True):
        self.metaData = metaData
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
            X = X.loc[ix]
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

def impute_NaN(data, label):
    data = data.dropna(subset=[label])
    # Impute NaN values separately for each class label
    for label_value in data[label].unique():
        class_data = data[data[label] == label_value]
        mean_values = np.ceil(class_data.mean(skipna=True))
        data.loc[data[label] == label_value] = class_data.fillna(mean_values)
    data = data.dropna(axis=1) # drop columns where no numeric values are present
    return data

def select_features_ttest(data, label, alpha):
    p_values = {}
    selected_features = [label]

    group_0 = data[data[label] == 0]
    group_1 = data[data[label] == 1]

    min_size = min(len(group_0), len(group_1))
    group_0 = group_0.sample(min_size)
    group_1 = group_1.sample(min_size)

    for feature in data.columns:
        if feature != label:
            # Perform a paired t-test
            t_stat, p_value = stats.ttest_ind(group_0[feature], group_1[feature])
            p_values[feature] = p_value

            if p_value < alpha:
                selected_features.append(feature)

    # Create a new DataFrame with only the selected features and the label
    selected_data = data[selected_features]
    print(f"Select {len(selected_features)-1} features with p_values smaller than {alpha}, original features: {len(data.columns)-1}")

    return selected_data, p_values

def select_features_importance(data, label, threshold=0.01):
    X = data.drop(label, axis=1)
    y = data[label]

    model = xgb.XGBClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    
    feature_importances = dict(zip(X.columns, importances))
    print(feature_importances)
    selected_features = [feature for feature, importance in feature_importances.items() if importance > threshold]
    selected_data = data[selected_features + [label]]
    
    return selected_data, feature_importances

def select_features_manual(data, file):
    with open(file, 'r') as f:
        feature_names = [line.strip() for line in f]

    # Filter the dataset to include only the features in the list and the label
    selected_features = [f for f in feature_names if f in data.columns]
    selected_data = data[selected_features]

    return selected_data

def select_by_wave(data, sample, feature):
    with open('../data/feature_selection.json', 'r') as file:
        feature_dict = json.load(file)
    with open('../data/sample_selection.json', 'r') as file:
        sample_dict = json.load(file)
    
    features = feature_dict[feature]
    samples = sample_dict[sample]
    data = data.iloc[samples]
    data = data[list(set(features).intersection(data.columns))]
    return data

def transform_2_oneWavePsample(data):
    Wave1 = ["INCA",   "INCE",   "INCF",   "INCD",   "INCB",   "INCC",   "INCG",   "W1BABS", "W1COLGRD", "W1LTHS", "W1ADVDEG", "W1COMHS", "W1SOMCOL", "W1OTHLNG", "CD5A",   "W1EDLEVL", "CD2A",   "CD4A_A",   "CD4A_B",   "CD4A_C",   "CD4A_D",   "CD4A_E",   "CD4A_F",   "X1",   "CD6A",   "W1CL1R", "W1COUNTY", "W1BRNUSA", "W1MALE", "W1FEMALE", "CC3",   "CD15",   "CD1",   "W1STRATA", "CD6",   "CD8",   "CD2"]
    Wave2 = ["W2INCA", "W2INCE", "W2INCF", "W2INCD", "W2INCB", "W2INCC", "W2INCG", "W2BABS", "W2COLGRD", "W2LTHS", "W2ADVDEG", "W2COMHS", "W2SOMCOL", "W2OTHLNG", "W2CD5A", "W2EDLEVL", "W2CD2A", "W2CD4A_A", "W2CD4A_B", "W2CD4A_C", "W2CD4A_D", "W2CD4A_E", "W2CD4A_F", "W2X1", "W2CD6A", "W2CL1R", "W2COUNTY", "W2BRNUSA", "W2MALE", "W2FEMALE", "W2CC3", "W2CD15", "W2CD1", "W2STRATA", "W2CD6", "W2CD8", "W2CD2"]
    Wave3 = ["W3INCA", "W3INCE", "W3INCF", "W3INCD", "W3INCB", "W3INCC", "W3INCG", "W3BABS", "W3COLGRD", "W3LTHS", "W3ADVDEG", "W3COMHS", "W3SOMCOL", "W3OTHLNG", "W3CD5A", "W3EDLEVL", "W3CD2A", "W3CD4A_A", "W3CD4A_B", "W3CD4A_C", "W3CD4A_D", "W3CD4A_E", "W3CD4A_F", "W3X1", "W3CD6A", "W3CL1R", "W3COUNTY", "W3BRNUSA", "W3MALE", "W3FEMALE", "W3CC3", "W3CD15", "W3CD1", "W3STRATA", "W3CD6", "W3CD8", "W3CD2"]
    Wave_Agnostic = ["AGE1829", "AGE3039", "AGE4049M", "AGE5059M", "AGE6064M", "AGE65PLM"]

    Generic_Wave = ["INCA",   "INCE",   "INCF",   "INCD",   "INCB",   "INCC",   "INCG",   "BABS", "COLGRD", "LTHS", "ADVDEG", "COMHS", "SOMCOL", "OTHLNG", "CD5A",   "EDLEVL", "CD2A",   "CD4A_A",   "CD4A_B",   "CD4A_C",   "CD4A_D",   "CD4A_E",   "CD4A_F",   "X1",   "CD6A",   "CL1R", "COUNTY", "BRNUSA", "MALE", "FEMALE", "CC3",   "CD15",   "CD1",   "STRATA", "CD6",   "CD8",   "CD2"]
    Generic_full_wave = Generic_Wave + Wave_Agnostic
    # This Code block creates a new data frame where the frist wave seen is taken from each sample
    # data = pd.read_csv("../data/ICPSR_36371/DS0001/36371-0001-Data.tsv", sep='\t')
    # for col in data:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').fillna(-1).astype(int)

    sample_wave = [0 for i in range(3661)]

    for i in range(3661):

        # 1, 2, 3
        if data["PANEL123"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV2"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 1

        # 1 and 3
        elif data["PANEL103"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 1

        # 1 and 2
        elif data["PANEL120"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV2"][i] == 1:
            sample_wave[i] = 1

        # 2 and 3
        elif data["PANEL023"][i] == 1 and data["ALLWAV2"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 2
        
        # 1
        elif data["ALLWAV1"][i] == 1:
            sample_wave[i] = 1

        # 2
        elif data["ALLWAV2"][i] == 1:
            sample_wave[i] = 2
        
        # 3
        elif data["ALLWAV3"][i] == 1:
            sample_wave[i] = 3

    # sample_wave: list of size 3661 which corrsponds to the first wave a sample has been seen
    data_n = []

    for i, val in enumerate(sample_wave):
        temp = []
        if val == 1:

            for j, var in enumerate(Wave1 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)

        elif val == 2:

            for j, var in enumerate(Wave2 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)

        elif val == 3:

            for j, var in enumerate(Wave3 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)
        
    data_new = pd.DataFrame(data=data_n, columns=Generic_full_wave)
    print(data_new.shape)
    return data_new


def transform_2_allWavePsample(data):
    # CL1R
    Wave1 = ["INCA",   "INCE",   "INCF",   "INCD",   "INCB",   "INCC",   "INCG",   "W1BABS", "W1COLGRD", "W1LTHS", "W1ADVDEG", "W1COMHS", "W1SOMCOL", "W1OTHLNG", "CD5A",   "W1EDLEVL", "CD2A",   "CD4A_A",   "CD4A_B",   "CD4A_C",   "CD4A_D",   "CD4A_E",   "CD4A_F",   "X1",   "CD6A",   "W1CL1R", "W1COUNTY", "W1BRNUSA", "W1MALE", "W1FEMALE", "CC3",   "CD15",   "CD1",   "W1STRATA", "CD6",   "CD8",   "CD2"]
    Wave2 = ["W2INCA", "W2INCE", "W2INCF", "W2INCD", "W2INCB", "W2INCC", "W2INCG", "W2BABS", "W2COLGRD", "W2LTHS", "W2ADVDEG", "W2COMHS", "W2SOMCOL", "W2OTHLNG", "W2CD5A", "W2EDLEVL", "W2CD2A", "W2CD4A_A", "W2CD4A_B", "W2CD4A_C", "W2CD4A_D", "W2CD4A_E", "W2CD4A_F", "W2X1", "W2CD6A", "W2CL1R", "W2COUNTY", "W2BRNUSA", "W2MALE", "W2FEMALE", "W2CC3", "W2CD15", "W2CD1", "W2STRATA", "W2CD6", "W2CD8", "W2CD2"]
    Wave3 = ["W3INCA", "W3INCE", "W3INCF", "W3INCD", "W3INCB", "W3INCC", "W3INCG", "W3BABS", "W3COLGRD", "W3LTHS", "W3ADVDEG", "W3COMHS", "W3SOMCOL", "W3OTHLNG", "W3CD5A", "W3EDLEVL", "W3CD2A", "W3CD4A_A", "W3CD4A_B", "W3CD4A_C", "W3CD4A_D", "W3CD4A_E", "W3CD4A_F", "W3X1", "W3CD6A", "W3CL1R", "W3COUNTY", "W3BRNUSA", "W3MALE", "W3FEMALE", "W3CC3", "W3CD15", "W3CD1", "W3STRATA", "W3CD6", "W3CD8", "W3CD2"]
    Wave_Agnostic = ["AGE1829", "AGE3039", "AGE4049M", "AGE5059M", "AGE6064M", "AGE65PLM"]

    Generic_Wave = ["INCA",   "INCE",   "INCF",   "INCD",   "INCB",   "INCC",   "INCG",   "BABS", "COLGRD", "LTHS", "ADVDEG", "COMHS", "SOMCOL", "OTHLNG", "CD5A",   "EDLEVL", "CD2A",   "CD4A_A",   "CD4A_B",   "CD4A_C",   "CD4A_D",   "CD4A_E",   "CD4A_F",   "X1",   "CD6A",   "CL1R", "COUNTY", "BRNUSA", "MALE", "FEMALE", "CC3",   "CD15",   "CD1",   "STRATA", "CD6",   "CD8",   "CD2"]
    Generic_full_wave = Generic_Wave + Wave_Agnostic

    sample_wave = [0 for i in range(len(data))]

    for i in range(len(data)):

        # 1, 2, 3
        if data["PANEL123"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV2"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 123

        # 1 and 3
        elif data["PANEL103"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 13

        # 1 and 2
        elif data["PANEL120"][i] == 1 and data["ALLWAV1"][i] == 1 and data["ALLWAV2"][i] == 1:
            sample_wave[i] = 12

        # 2 and 3
        elif data["PANEL023"][i] == 1 and data["ALLWAV2"][i] == 1 and data["ALLWAV3"][i] == 1:
            sample_wave[i] = 23
        
        # 1
        elif data["ALLWAV1"][i] == 1:
            sample_wave[i] = 1

        # 2
        elif data["ALLWAV2"][i] == 1:
            sample_wave[i] = 2
        
        # 3
        elif data["ALLWAV3"][i] == 1:
            sample_wave[i] = 3

    # sample_wave: list of size 3661 which corrsponds to the first wave a sample has been seen
    data_n = []

    for i, val in enumerate(sample_wave):
        temp = []
        if val == 123:
            for j, var in enumerate(Wave1 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)
            temp = []

            for j, var in enumerate(Wave2 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)
            temp = []

            for j, var in enumerate(Wave3 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)

        elif val == 12:
            for j, var in enumerate(Wave1 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)
            temp = []

            for j, var in enumerate(Wave2 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)

        elif val == 13:
            for j, var in enumerate(Wave1 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)
            temp = []

            for j, var in enumerate(Wave3 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)

        elif val == 23:
            for j, var in enumerate(Wave2 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)
            temp = []

            for j, var in enumerate(Wave3 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)

        elif val == 1:

            for j, var in enumerate(Wave1 + Wave_Agnostic):
                temp.append(data[var][i])

            data_n.append(temp)

        elif val == 2:

            for j, var in enumerate(Wave2 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)

        elif val == 3:

            for j, var in enumerate(Wave3 + Wave_Agnostic):
                temp.append(data[var][i])
            
            data_n.append(temp)
        
    data_new = pd.DataFrame(data=data_n, columns=Generic_full_wave)
    print(data_new.shape)
    return data_new

def reduce_data(data, pc_reducer, umap_reducer, args):
    if args.d == "pca":
        data = pc_reducer.fit_transform(data)
    elif args.d == "umap":
        data = umap_reducer.fit_transform(data)
    elif args.d == "pca_umap":
        data = pc_reducer.fit_transform(data)
        data = umap_reducer.fit_transform(data)
    return data

def encode_label(data, label):
    data[label].fillna(0, inplace=True)
    data[label] = data[label].apply(lambda x: 1 if x in [1.0, 2.0] else 0)

    # data = data.dropna(subset=[label])
    # data = data[data[label] != 9]
    # data[label] = data[label].apply(lambda x: 1 if x in [1.0, 2.0] else 0)

    return data

def make_data(data, label, configs, args, type, dataCleaner = None, feature_selector = None, pc_reducer = None, umap_reducer = None):
    if type == "training":
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
        
        pc_reducer = Pipeline([("pca", pcaReducer(criteria=configs["DataProcessing"]["feature_selection"]["pcaCriteria"]))])
        umap_reducer = Pipeline([("umap", umapReducer(seed=configs["seed"]))])

        data = feature_selector.fit_transform(data)
        training_indices = np.load(os.path.join(configs["outputDir"], 'training_indices.npy'))
        data.index = training_indices
        target = target.loc[training_indices, :]
        if not args.d == "raw":
            data = reduce_data(data, pc_reducer, umap_reducer, args)
        return data, target, dataCleaner, feature_selector, pc_reducer, umap_reducer
    else:
        report = audit_data(data, duplicateRow = configs["DataProcessing"]["audit"]["duplicateRow"], duplicateCol = configs["DataProcessing"]["audit"]["duplicateCol"], 
                        NaN = configs["DataProcessing"]["audit"]["NaN"], column_NaN = configs["DataProcessing"]["audit"]["column_NaN"],  
                        maxNA = configs["DataProcessing"]["audit"]["maxNA"])
        # data cleaning
        print(f"{'*'*30} Data Cleaning {'*'*30}")
        dataCleaner.set_params(NaColumnsHandler__report=report)
        dataCleaner.set_params(NaHandler__report=report)
        dataCleaner.set_params(duplicateHandler__report=report)
        dataCleaner.set_params(NaHandler__evaluation=True)
        dataCleaner.set_params(duplicateHandler__evaluation=True)
        dataCleaner.set_params(rowOutlierChecker__evaluation=True)
        data = dataCleaner.fit_transform(data)

        target = data[[label]]
        data = data.drop(columns=[label])

        print(f"{'*'*30} Select Features {'*'*30}")
        # features processing
        feature_selector.set_params(dataBalancer__training=False)
        data = feature_selector.transform(data)
        if not args.d == "raw":
            data = reduce_data(data, pc_reducer, umap_reducer, args)
        return data, target
