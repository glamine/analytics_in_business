# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:55:12 2019

@author: user
"""

# =============================================================================
# import pandas as pd 
# 
# data = pd.read_csv("merged4.csv")
# 
# =============================================================================
# Importing libraries
import pandas as pd
import numpy as np

# Read csv file into a pandas dataframe
df = pd.read_csv("merged4.csv")

# Take a look at the first few rows
print(df.head())