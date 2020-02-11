# %% Imports and file load
import pathlib, numpy as np, pandas as pd, seaborn as sns, matplotlib, matplotlib.pyplot as plt

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd() / 'CaseCompetitionData' / 'CaseCompetitionData.csv'
data = pd.read_csv(datafolder)

data['customer_age'] = data['customer_age'].map({'0 - 20':1,
                                                '21 - 30':2,
                                                '31 - 40':3,
                                                '41 - 50':4,
                                                '51 - 60':5,
                                                '61 - 70':6,
                                                '71 - 80':7,
                                                '81 - 90':8,
                                                '91 - 100':9,
                                                '101+':10,
                                                '?':np.nan})
data['customer_income'] = data['customer_income'].map({'0 - 20000': 1,
                                                    '20001 - 40000': 2,
                                                    '40001 - 60000': 3,
                                                    '60001 - 80000': 4,
                                                    '80001 - 100000': 5,
                                                    '100001 - 120000': 6,
                                                    '120001 - 140000': 7,
                                                    '140001 - 160000': 8,
                                                    '160001 - 180000': 9,
                                                    '180001 - 200000': 10,
                                                    '200001+':11,
                                                    '?':np.nan})
data['purchase_price'] = data['purchase_price'].map({'0 - 5000':1,
                                                    '5001 - 10000':2,
                                                    '10001 - 15000':3,
                                                    '15001 - 20000':4,
                                                    '20001 - 25000':5,
                                                    '25001 - 30000':6,
                                                    '30001 - 35000':7,
                                                    '35001 - 40000':8,
                                                    '40001 - 45000':9,
                                                    '45001 - 50000':10,
                                                    '50001 - 55000':11,
                                                    '55001 - 60000':12,
                                                    '60001 - 65000':13,
                                                    '65001 - 70000':14,
                                                    '70001 - 75000':15,
                                                    '75001 - 80000':16,
                                                    '80001 - 85000':17,
                                                    '85001 - 90000':18,
                                                    '90001 - 95000':19,
                                                    '?':np.nan})

data.loc[data.customer_distance_to_dealer == '?', 'customer_distance_to_dealer'] = np.nan
data['customer_distance_to_dealer'] = data['customer_distance_to_dealer'].astype('float64')

data['loyal'] = 0
data.loc[data.subsequent_purchases > 0, 'loyal'] = 1

# %% Determining how to handle missing values

# Missing values for purchase price are such a small percentage, they can be dropped
data.groupby(pd.isnull(data['purchase_price']))['loyal'].count()
data.dropna(subset=['purchase_price'],inplace=True)

# Missing values for age are small but might affect response, so they'll be filled
# with median and have a column added to indicate a missing value
data.groupby(pd.isnull(data['customer_age']))['loyal'].count()
data.groupby(pd.isnull(data['customer_age']))['loyal'].mean()
data.groupby('loyal')['customer_age'].value_counts()
data['age_missing'] = 0
data.loc[pd.isnull(data.customer_age), 'age_missing'] = 1
data['customer_age'].fillna(data['customer_age'].median(), inplace=True)

# Same deal with missing values for distance to dealer
data.groupby(pd.isnull(data.customer_distance_to_dealer))['loyal'].count()
data.groupby(pd.isnull(data.customer_distance_to_dealer))['loyal'].mean()
data.groupby('loyal')['customer_distance_to_dealer'].median()
data['distance_missing'] = 0
data.loc[pd.isnull(data.customer_distance_to_dealer), 'distance_missing'] = 1
data['customer_distance_to_dealer'].fillna(data['customer_distance_to_dealer'].median(), inplace=True)

# Missing values for income are fairly large. They don't affect the mean of the
# response but they are disproportionately missing for other factors. They'll be
# filled with median and have a column added to indicate missing values
data.groupby(pd.isnull(data['customer_income']))['loyal'].count()
data.groupby(pd.isnull(data['customer_income']))['loyal'].mean()
data['income_missing'] = 0
data.loc[pd.isnull(data.customer_income), 'income_missing'] = 1
data['customer_income'].fillna(data['customer_income'].median(), inplace=True)

# Too many values are missing for post purchase satisfaction, so the whole feature
# will replaced with whether or not it was completed
data.groupby(data['post_purchase_satisfaction']=='?')['loyal'].count()
data['survey_missing'] = 0
data.loc[data.post_purchase_satisfaction == '?','survey_missing'] = 1
data.drop(columns=['post_purchase_satisfaction'], inplace=True)

# %%
data.dtypes
