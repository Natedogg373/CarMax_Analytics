# %% Imports and file load
import pathlib, math, numpy as np, pandas as pd, seaborn as sns, matplotlib, matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, mutual_info_regression, RFECV
from sklearn.feature_extraction import FeatureHasher
from sklearn import metrics, preprocessing
from scipy import stats

le_make = preprocessing.LabelEncoder()
le_model = preprocessing.LabelEncoder()
le_cat = preprocessing.LabelEncoder()
oh_make = preprocessing.OneHotEncoder()
oh_model = preprocessing.OneHotEncoder()

sns.set()
%matplotlib inline
pd.options.mode.chained_assignment = None

datafolder = pathlib.Path.cwd() / 'CaseCompetitionData' / 'CaseCompetitionData.csv'
data = pd.read_csv(datafolder)

luxury = ['AUDI','BENTLEY','BMW','CADILLAC','INFINITI','JAGUAR','LAND ROVER','LEXUS','MASERATI','MERCEDES-BENZ','PORSCHE']
midrange = ['ACURA','BUICK','CHEVROLET','DODGE','FORD','GMC','HUMMER','JEEP','LINCOLN','LOTUS','MINI','SUBARU','TOYOTA','VOLVO','VOLKSWAGEN']
lowend = ['CHRYSLER','FIAT','HONDA','HYUNDAI','KIA','MAZDA','MERCURY','MITSUBISHI','NISSAN','PONTIAC','SCION','SUZUKI']
budget = ['DAEWOO','EAGLE','GEO','ISUZU','OLDSMOBILE','PLYMOUTH','SAAB','SATURN','SMART']

make_popularities = data.purchase_make.value_counts().to_frame().reset_index()
model_popularities = data.purchase_model.value_counts().to_frame().reset_index()
make_popularities.rename(columns={'purchase_make':'make_popularity'},inplace=True)


data = data.merge(make_popularities, how='left', left_on='purchase_make', right_on='index')
data.drop(columns=['index'],inplace=True)

data.loc[data.customer_distance_to_dealer == '?', 'customer_distance_to_dealer'] = np.nan
data['customer_distance_to_dealer'] = data['customer_distance_to_dealer'].astype('float64')

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
data['customer_gender'] = data['customer_gender'].map({'M':0,'F':1,'U':np.nan})

data['loyal'] = 0
data.loc[data.subsequent_purchases > 0, 'loyal'] = 1

data['purchase_make_cat'] = np.nan
data.loc[data['purchase_make'].isin(luxury),'purchase_make_cat'] = 'Luxury'
data.loc[data['purchase_make'].isin(midrange),'purchase_make_cat'] = 'Mid-Range'
data.loc[data['purchase_make'].isin(lowend),'purchase_make_cat'] = 'Low-End'
data.loc[data['purchase_make'].isin(budget),'purchase_make_cat'] = 'Budget'


# %% Run if numerics needed

data['purchase_make'] = le_make.fit_transform(data.purchase_make)
data['purchase_model'] = le_model.fit_transform(data.purchase_model)
data['purchase_make_cat'] = le_cat.fit_transform(data.purchase_make_cat)

# %% Determining how to handle missing values

# Missing values for purchase price are such a small percentage, they can be dropped
#data.groupby(pd.isnull(data['purchase_price']))['loyal'].count()
data.dropna(subset=['purchase_price'],inplace=True)

# Missing values for age are small but might affect response, so they'll be filled
# with median and have a column added to indicate a missing value
#data.groupby(pd.isnull(data['customer_age']))['loyal'].count()
#data.groupby(pd.isnull(data['customer_age']))['loyal'].mean()
#data.groupby('loyal')['customer_age'].value_counts()
#data['age_missing'] = 0
#data.loc[pd.isnull(data.customer_age), 'age_missing'] = 1
data['customer_age'].fillna(data['customer_age'].median(), inplace=True)

# Same deal with missing values for distance to dealer
#data.groupby(pd.isnull(data.customer_distance_to_dealer))['loyal'].count()
#data.groupby(pd.isnull(data.customer_distance_to_dealer)).mean()
#data.groupby('loyal')['customer_distance_to_dealer'].median()
data['distance_missing'] = 0
data.loc[pd.isnull(data.customer_distance_to_dealer), 'distance_missing'] = 1
#data['customer_distance_to_dealer'].fillna(data['customer_distance_to_dealer'].median(), inplace=True)
s = data.customer_distance_to_dealer.value_counts(normalize=True)
data.loc[data.distance_missing == 1,'customer_distance_to_dealer'] = np.random.choice(s.index, size=len(data.loc[data.distance_missing == 1,'customer_distance_to_dealer']),p=s.values)
distance_bins = [-1,.1, 1.1, 5.1, 10.1, 25.1, 50.1, 100.1, 2500.1]
distance_labels = [1,2,3,4,5,6,7,8]
data['distance_binned'] = pd.cut(data['customer_distance_to_dealer'], bins=distance_bins, labels=distance_labels)

def randomboolean(rand):
    if rand > 0.6:
        return 1;
    else:
        return 0;
# Same deal with missing values for gender
data.groupby(pd.isnull(data.customer_gender))['loyal'].count()
#data.groupby(pd.isnull(data.customer_gender))['loyal'].mean()
#data.groupby('loyal')['customer_gender'].mean()
data['gender_missing'] = 0
data.loc[pd.isnull(data.customer_gender), 'gender_missing'] = 1
data['customer_gender'].fillna(randomboolean(np.random.rand()), inplace=True)

# Missing values for income are fairly large. They don't affect the mean of the
# response but they are disproportionately missing for other factors. They'll be
# filled with median and have a column added to indicate missing values
#data.groupby(pd.isnull(data['customer_income']))['loyal'].count()
#data.groupby(pd.isnull(data['customer_income']))['loyal'].mean()
data['income_missing'] = 0
data.loc[pd.isnull(data.customer_income), 'income_missing'] = 1
data['customer_income'].fillna(data['customer_income'].median(), inplace=True)

# Too many values are missing for post purchase satisfaction, so the whole feature
# will replaced with whether or not it was completed
#data.groupby(data['post_purchase_satisfaction']=='?')['loyal'].count()
#data['survey_missing'] = 0
#data.loc[data.post_purchase_satisfaction == '?','survey_missing'] = 1
data.drop(columns=['post_purchase_satisfaction'], inplace=True)


# %% Separate Feature Sets
cat_features = data[['purchase_make','purchase_model','customer_gender','purchase_make_cat','trade_in','vehicle_financing','distance_missing','income_missing','gender_missing','customer_previous_purchase','vehicle_warranty_used','distance_binned']]
cat_label = data['loyal']

num_features = data[['purchase_vehicle_year','purchase_price','customer_age','customer_income','customer_distance_to_dealer','make_popularity']]
num_label = data['subsequent_purchases']

################################################################################
# UNIVARIATE FEATURE SELECTION
################################################################################

# %% chi2
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(cat_features,cat_label)
dfscores = pd.DataFrame(fit.scores_)
dfpvalues = pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(cat_features.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
featureScores.columns = ['Specs','Score','Pval']  #naming the dataframe columns\
featureScores.sort_values('Score',ascending=False)

sns.heatmap(data.corr().abs(),square=True)

sns.scatterplot(data=data,x='subsequent_purchases',y='customer_income')

for i in ['purchase_vehicle_year','purchase_price','customer_age','customer_income','customer_distance_to_dealer','make_popularity']:
    print(i)
    print(stats.kendalltau(data[i],data['subsequent_purchases']))

for i in ['purchase_make','purchase_model','customer_gender','purchase_make_cat','trade_in','vehicle_financing','distance_missing','income_missing','gender_missing','customer_previous_purchase','vehicle_warranty_used','distance_binned']:
    print(i)
    groupednumbers = {}
    for grp in data[i].unique():
        groupednumbers[grp] = data['loyal'][data[i]==grp].values
    args = groupednumbers.values()

    print(stats.kruskal(*args))



###############################################################################
# MACHINE LEARNING MODEL SELECTION
###############################################################################

# %% RandomForestClassifier
prefeatures = data.drop(columns=['loyal','subsequent_purchases','insert_num','distance_missing'])
postfeatures = prefeatures.drop(columns=['gender_missing','income_missing','purchase_make_cat','customer_age'])
postfeatures.columns
labels = data['loyal']

# %%
features = postfeatures
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

rf = RandomForestClassifier(n_estimators=30, max_depth=5)
rf.fit(X_train, y_train)
scores = rf.score(X_val, y_val)
predictions = rf.predict(X_test)
metrics.classification_report(y_test, predictions, labels=[1,0])
rf.feature_importances_
X_test.columns
metrics.accuracy_score(y_test, predictions)

data.purchase_model.unique()

# %%
features = postfeatures
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model = RidgeClassifier()
model.fit(X_train, y_train)
scores = model.score(X_val, y_val)
predictions = model.predict(X_test)
metrics.classification_report(y_test, predictions, labels=[1,0])
metrics.accuracy_score(y_test, predictions)

# %%

'''
# %% Regressor test
features = data.drop(columns=['loyal','purchase_make','purchase_model','subsequent_purchases','distance_missing','customer_distance_to_dealer','insert_num'])
labels = data['subsequent_purchases']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

rf = RandomForestRegressor(n_estimators=50, max_depth=10)
rf.fit(X_train, y_train)
scores = rf.score(X_val, y_val)
scores
predictions2 = rf.predict(X_test)
metrics.mean_squared_error(y_test, predictions2)
rf.feature_importances_
X_test.columns
metrics.confusion_matrix(y_test, predictions2, labels=[1,0])
'''

###############################################################################

# %%
rf = RandomForestClassifier(n_estimators=30,max_depth=10)
lr = LogisticRegression(solver='lbfgs')
bestfeatures = RFECV(rf)
fit = bestfeatures.fit(features,labels)
dfscores = pd.DataFrame(fit.ranking_)
dfcolumns = pd.DataFrame(features.columns)
supp = pd.DataFrame(fit.support_)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores,supp],axis=1)
featureScores.columns = ['Specs','Score','Support']  #naming the dataframe columns
featureScores.sort_values('Score',ascending=True)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(bestfeatures.grid_scores_) + 1), bestfeatures.grid_scores_)
plt.show()

# %%
rf = RandomForestClassifier()
lr = LogisticRegression(solver = 'sag')
bestfeatures = RFECV(rf)
fit = bestfeatures.fit(features,num_label)
dfscores = pd.DataFrame(fit.ranking_)
dfcolumns = pd.DataFrame(features.columns)
supp = pd.DataFrame(fit.support_)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores,supp],axis=1)
featureScores.columns = ['Specs','Score','Support']  #naming the dataframe columns
featureScores.sort_values('Score',ascending=True)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(bestfeatures.grid_scores_) + 1), bestfeatures.grid_scores_)
plt.show()

################################################################################
################################################################################










# %%
sns.set()
sns.set_context("talk")
for feature in data.columns.values.tolist():
    fig, axs = plt.subplots(ncols=2,figsize=(15,10))
    sns.violinplot(x='loyal',y=feature, data=data, ax=axs[0])
    sns.boxplot(x='loyal',y=feature, data=data, ax=axs[1])
    plt.show()
fig = sns.pairplot(data=test_set, hue='loyal', kind='reg')

data.groupby(['customer_income','loyal'])['insert_num'].count().to_csv('output.csv')
sns.countplot(x='customer_previous_purchase',data=data,hue='loyal')

sns.heatmap(data.pivot_table(index='purchase_price',columns=['customer_age'],values='loyal',aggfunc='mean'),cmap='Blues',square=True)
data.pivot_table(index='customer_income',columns=['purchase_price'],values='loyal',aggfunc='mean').to_csv('output.csv')
popularities.to_csv('output.csv')

data.pivot_table(index='make_popularity',values='customer_distance_to_dealer', aggfunc='mean').to_csv('output.csv')
data.groupby('subsequent_purchases').mean()
