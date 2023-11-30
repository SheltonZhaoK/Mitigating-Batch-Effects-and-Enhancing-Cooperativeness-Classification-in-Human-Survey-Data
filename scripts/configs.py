# -----------------------------------------------------------
# This script provides hyperparameter used in the experiments
#
# Author: Konghao Zhao
# Created: 2023-10-27
# Modified: 2023-10-30
# 
# -----------------------------------------------------------
class Configs:
    def __init__(self):
        self.configs = {
            "seed": 1,
            "test_size": 0.3,
            
            "DataProcessing":
            {
                "audit":
                {
                    "duplicateRow": True,
                    "duplicateCol": True,
                    "NaN": True,
                    "column_NaN": True,
                    "maxNA": 0.52
                },

                "cleaning":
                {
                    "stdScale": 3,
                    "evaluation": False
                },

                "feature_selection":
                {
                    "correlationUpperBound": 0.98,
                    "scaleRange": [-1, 1],
                    "numVariableFeatures": 100,
                    "pcaCriteria":0.99999999
                    # "pcaCriteria":30
                },
            }
    
        }