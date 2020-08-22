# --------Imports---------
import pandas as pd
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# Integer encoding
from sklearn.preprocessing import OneHotEncoder

# Imputing for missing values
from sklearn.impute import SimpleImputer

# Dimensionality reduction
from sklearn.decomposition import PCA

# Scaling data
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Standardization, Normalization

# Train Test Split
from sklearn.model_selection import train_test_split

# Imbalance data
from imblearn.over_sampling import SMOTE


# ----------------------------


# ------------Cleaning Data-----------------
# Bank data
bank_df = pd.read_csv("Data/bank.csv")

# Removing Loan ID and Customer ID columns and Years of Credit History
bank_df.drop(columns=["Years of Credit History"], axis=1, inplace=True)

# Replacing Credit Score column to strings
bank_df["Credit Score"] = bank_df["Credit Score"].astype("str")

# Removing periods and 0s from the Credit Score column
bank_df["Credit Score"] = [nums.replace("0", "") for nums in bank_df["Credit Score"]]
bank_df["Credit Score"] = [nums.replace(".", "") for nums in bank_df["Credit Score"]]

# Converting Credit Score column from a str to a float
bank_df["Credit Score"] = bank_df["Credit Score"].astype("float64")

# Creating max and min cap locks to remove outliers
def max_cap_lock(df, column:str, cap:int):
    index = df[column].loc[df[column] >= cap].index
    df.drop(index=index, inplace=True)
    
def min_cap_lock(df, column:str, cap:int):
    index = df[column].loc[df[column] <= cap].index
    df.drop(index=index, inplace=True)
	
	
max_cap_lock(bank_df, "Current Loan Amount", 1_000_000)
max_cap_lock(bank_df, "Number of Open Accounts", 35)
max_cap_lock(bank_df, "Monthly Debt", 150_000)
max_cap_lock(bank_df, "Months since last delinquent", 80)
max_cap_lock(bank_df, "Number of Credit Problems", 6)
max_cap_lock(bank_df, "Bankruptcies", 5)
max_cap_lock(bank_df, "Tax Liens", 5)

# Numeric data
num_var = bank_df.select_dtypes(exclude="object")
# Separating categorical from numeric variables
cat_variables = bank_df.select_dtypes(include="object")
cat_variables.drop(columns=["Loan ID", "Customer ID"], inplace=True)  # Dropping redundant columns

# Using get dummies to convert categorical values to numeric
cat_to_num = pd.get_dummies(cat_variables, drop_first=True)  # Performing get dummies on categorical values

# Using SimpleImputer with a mean strategy to fill missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

impute_num = imputer.fit_transform(num_var)  # Imputing missing values for numeric data
impute_cat = imputer.fit_transform(cat_to_num)  # Imputing missing values for encoded data
impute_num_df = pd.DataFrame(impute_num, columns=num_var.columns)  # Changing the array into a dataframe for numeric
impute_cat_df = pd.DataFrame(impute_cat, columns=cat_to_num.columns)  # Converting array to a dataframe for cat_to_num (encoded values)

# Combining dataframes
bank_df = impute_num_df.join(impute_cat_df, how="left")

# Dropping duplicated data
bank_df.drop_duplicates(inplace=True)

#----------------------------------------------


# -------------------------PCA------------------------------
# Independent and Dependent variables
X = bank_df.drop(columns="Loan Status_Fully Paid", axis=1)
y = bank_df["Loan Status_Fully Paid"]

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# Intantiating Synthetic Minority Over Sampling Technique to balance target variable
sm = SMOTE(random_state=19)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train)

# Normalizing data
min_max_scaler = MinMaxScaler() # Values will be put in a range from 0 to 1
# Using normalization because the machine learning models that will be used don't have any assumptions/requirements of the data being in a normal distribution
# also some of our data are not in a Gaussian Distribution (Bell Curve/Normal Distribution)

# Normalizing float64 columns to be put in a range of 0 to 1
col_rescaled = min_max_scaler.fit_transform(X_train_new)
X_train_rescaled = pd.DataFrame(data=col_rescaled, columns=X_train_new.columns)

# Using PCA for dimension reduction
pca = PCA(n_components=0.99, random_state=18)  # 99% variance/threshold
reduced_pca = pca.fit_transform(X_train_rescaled, y_train)

# Creating new dataset
X_train_pca = pd.DataFrame(data=reduced_pca)
# Adding PC as a prefix
X_train_pca = X_train_pca.add_prefix("PC")

#-------------------------------------------------------


