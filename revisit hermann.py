# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:06:54 2020

@author: acdav
"""
#NORMAL PYTHON TOOLS
import numpy as np
import pandas as pdd
import modin.pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import scipy as sp
import sklearn as sk

#IMPORT FOR EDA
import re

#VISUALIZATION TOOLS
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import missingno as msno

#MACHINE LEARNING AND SUPERVISED LEARNING TOOLS
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Evaluate
from xgboost import plot_importance
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

#Importing tools
from functools import partial
import sklearn.preprocessing
import statsmodels.formula.api as smm
import statsmodels.api as sm
import shap
import os
from functools import partial

#helping with memory
import tensorflow as tf
from keras import backend as k

#LOOKING FOR THE BEST HYPERPARAMETERS TOOLS
import parfit as pf
from parfit import bestFit # Necessary if you wish to use bestFit
# Necessary if you wish to run each step sequentially
from parfit.fit import *
from parfit.score import *
from parfit.plot import *
from parfit.crossval import *

%matplotlib inline

#Change my working directory to more easily locate my data files.
os.chdir(r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Spyder projects\research projects')

#View my working directory.
os.getcwd()

#Import the data from Memorial Hermann Hospital.
df = pd.read_excel(r'PEDI1116ALL.xlsx')

#complete some initial EDA of the revised data.
df.info()
columns = df.columns
columns
df.head()

#create a copy of the data to be able to work with it so we dont overwrite it.
df1 = df.copy()
df1.info()

#We can see that we have some missing values so lets better visualize them. An since we have them all over lets look at them by percentage to better judge how to handle them.
percent_missing = df1.isnull().sum()* 100 / len(df1)
percent_missing

#Now lets sort the missing percentages and view the data.
missing_value_df1 = percent_missing.sort_values(ascending=False)
missing_value_df1

#Next lets drop the features that have a percentage of missing values under 39% (40%) and identify those columns.
missing = df1.loc[:, (percent_missing > 40)]
missing.info()

#Now lets drop the features that have a percentage of missing values over 39% (40%) and identify those columns that we should not have issues with. This will be the data that we use moving forward.
df2 = df1.loc[:, (percent_missing < 40)]
df2.info()

#Now lets look at the missing percentages of the columns that we kept to determine how we will handle those missing values.
percent_missing2 = df2.isnull().sum()* 100 / len(df2)
percent_missing2

#Now lets sort the missing percentages and view the data.
missing_value_df2 = percent_missing2.sort_values(ascending=False)
dfkeep = missing_value_df2
dfkeep

#Based on what we can see in remaining data we will want to drop the AIS features since they are captured in the GCS values.
df3 = df2.drop(['AIS_HEAD', 'AIS_EXTERN'], axis=1)
df3.info()

# Change the data type of ED_DBP for the next conversions to work well.
# We will fill the string values with NaN and then proceed.
df3['ED_DBP'] = df3['ED_DBP'].replace('UNK', np.nan)
df3['ED_DBP'] = df3['ED_DBP'].replace(999, np.nan)
df3['ED_DBP'] = df3['ED_DBP'].replace('P', np.nan)
df3['ED_HR'] = df3['ED_HR'].replace('UNK', np.nan)
df3['ED_DBP'] = df3['ED_DBP'].astype('float16')
df3['ED_HR'] = df3['ED_HR'].astype('float16')
df3['ED_DBP'].unique()
#Now we will convert the 5 columns with lots of missing values into categories.
a = pd.cut(df3['EDMAP'],[0,65,110,400],labels=['Dangerous','Normal range','Too High'])
b = pd.cut(df3['ED_DBP'],[0,65,90,300],labels=['Low','Normal Range','High'])
c = pd.cut(df3['ED_SBP'],[0,60,120,370],labels=['Low','Normal Range','High'])
d = pd.cut(df3['ED_HR'],[0,60,90,500],labels=['Low','Normal Range','High'])
e = pd.cut(df3['ED_GCS'],[0,8,12,15],labels=['Dangerous','Concern','No Concern'])

# Then insert the new categories next to their original rows.
df3.insert(14,'ED_MAP1',a)

df3.insert(15,'ED_DBP2',b)

df3.insert(16,'ED_SBP2',c)

df3.insert(17,'ED_HR2',d)

df3.insert(18,'ED_GCS1',e)

#Lets look to see if our new columns got added
df3.info()

# Next lets drop all of the duplicate, identifying, and irrelevant numbers, based on domain knowledge, to the dataset moving forward.
df4 = df3.drop(['ETHNICITY', 'MRN', 'PT_NUMB', 'DOB', 'INJ_DATE', 'INJ_TIME', 'NOTIFY_DT', 'SEX', 'TRAUMA_TYPE', 'TRANSPORT', 'PH_SBP', 'PH_DBP', 'PH_HR', 'PH_EYE', 'PH_VERB', 'PH_MOTOR', 'TRANSFER_', 'PH_GCS', 'NOTIFY_TM', 'RACE', 'PH_INTUBATION', 'HOSP_ARRIV_TIME', 'ED_SBP','ED_DBP', 'ED_HR','ED_EYE', 'ED_VERB', 'ED_MTR', 'ED_GCS', 'ED_INTUBATION', 'FLUIDS_TYPE', 'FLUIDS_LOC', 'Fluidloc', 'FLUIDS_AMT', 'OUTCOME', 'D_C_DATE', 'D_C_TIME', 'D_C_DISPO', 'age1', 'PHMAP', 'EDMAP', 'newarriv', 'PH_SBP1', 'PH_DBP1', 'PH_HR1', 'ht3', 'wt3', 'iss2', 'wt2', 'ht2', 'hr2', 'map2', 'sbp2', 'race2'], axis=1)

# Now lets look at the smaller more concise dataset.
df4.info()

# Next lets look into the data types that we will need in order to move forward. We will also start to look for missing values and take care of them along the way.
df4.head()

# Lets make a dictionary for the numeric/continuous, floats, and one for the categorical features to make this easier. Also, lets see why HEIGHT and WEIGHT are objects. There may be a character value in the columns. If there is we will change it to na.
df4['HEIGHT'].unique()
# We found that there are some character inside the row and I imagine the same goes for WEIGHT. so we will convert these to nan.
df4['HEIGHT'] = df4['HEIGHT'].replace('UNK', np.nan)
df4['WEIGHT'] = df4['WEIGHT'].replace('UNK', np.nan)
df4['BMI'] = df4['BMI'].replace('UNK', np.nan)

catnumflo = {'ED_MAP1':'category','ED_DBP2':'category','ED_SBP2':'category','ED_HR2':'category','ED_GCS1':'category', 'ICU_TOTAL':'uint8', 'VENT':'uint8', 'ISS':'uint8', 'out':'uint8', 'sex1':'uint8', 'FASTCOM':'uint8', 'TMAT':'uint8', 'race1':'uint8', 'Transfer':'uint8', 'Fluidtype':'uint8', 'BLDPGVN':'uint8', 'FASTCOMPOS':'uint8', 'FASTCOMNEG':'uint8', 'FASTNOTCOM':'uint8', 'SHI':'uint8', 'CondHI':'uint8', 'HEIGHT':'float32', 'WEIGHT':'float32', 'AGE':'float16', 'TransT':'float16', 'LOS':'float16', 'ETHN':'float16', 'INTN':'float16', 'BMI':'float16'}

# Next we will change the data types to help us with analysis and more feature engineering that we will do in the coming steps.
df4 = df4.astype(catnumflo)

# Now lets visualize our data and also look for correlation to then boot values if we need to. Then we will handle the last missing values, and then see about onehot encoding.
df4.info()

# A correlation matrix will help us look for multicollinearity so we can take those features out. We will look at spearman due to the data being imbalanced, having categorical data, and not being normally distributed.
df4corr = df4.corr(method='spearman')

# Lets visualize it.
fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(df4corr, xticklabels = df4corr.columns, yticklabels = df4corr.columns, annot = True,
           annot_kws={"size": 20}, ax=ax, vmin=-1, vmax=1, center=0,)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

# After visualizing we have some things that make sense. we will remove the VENT feature due to multicollinearity with the intubation and the ICU, and LOS features. We will also remove FASTNOTCOM and FASTCOM due to their multicollinearity with FAST competed and negative or positive, it is just to opposite of this feature. We will also take out BMI because it is just a combination of height and weight which we have. There was also multicollinearity between the fluid types and the blood given due to them all being blood products and crystalloid being associated more with patients not receiving blood. Race and ethnicity are also very correlated since they contain some of the same features. Finally we will take out ICU stay because this is ultimately included in the LOS. We will remove these features and then run the matrix again just to make sure we have not created other problems.
df5 = df4.drop(['VENT', 'FASTNOTCOM', 'FASTCOM', 'BMI', 'Fluidtype', 'ETHN', 'ICU_TOTAL', 'LOS'], axis=1)
df5.info()

# Lets run the correlation matrix again.
df4corr = df5.corr(method='spearman')

# Lets visualize it.
fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(df4corr, xticklabels = df4corr.columns, yticklabels = df4corr.columns, annot = True,
           annot_kws={"size": 20}, ax=ax, vmin=-1, vmax=1, center=0,)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

# We seem to be in better shape now. We will now move on to see what are our remaining missing value percentages before moving to modeling.
percent_missing5 = df5.isnull().sum()* 100 / len(df5)
percent_missing5
missing_value_df5 = percent_missing5.sort_values(ascending=False)
missing_value_df5

# Since our data is real, and what is missing is personal data that was not collected, we do not want to impute or replace given that these may skew the results and harm the integrity of the data. We will instead, drop all of the patients with missing values.
df6 = df5.dropna()
df6.info()

# Now lets look at the distribution of the target feature to see if we may need to do anything with the data.
sns.distplot(df6['BLDPGVN'])

# It does look like we have an imbalanced target, this will dictate the eval metric and models we use. We will want to use AUC and F1 score for a classification problem. We are going to use models that can handle mixed data as well as we are going to onehot encode to get a more accurate look at the prediction numbers. Next we will be onehot encoding a few features to include code level, EDMAP, EDDBP, EDSBP, EDHR, and GCS to get more understanding around the values. We will be label encoding the cause codes, and the transport agencies as well.

# Lets create the dataframes we will need to do this.
df66 =['CODE_LEVEL','ED_MAP1', 'ED_DBP2', 'ED_SBP2', 'ED_HR2', 'ED_GCS1', 'CAUSE_CODE']

# Onehot encode first.
df7 = pd.get_dummies(df6, prefix_sep='_', drop_first=False, columns = df66)
df7.info()

# Label encode next.
le = LabelEncoder()
df7['TRANS_AGENCY'] = le.fit_transform(df7['TRANS_AGENCY'])
df7['TRANS_AGENCY'].head(10)

# Now that we have processed our labels and made everything numeric we can proceed with creating our test and train set and modeling. We will be using pafit for our parameter tuning because we have a time element, which will cause us to split the data on that time element for prediction after sorting first. We will then create a train-validation-test set from the data so that we can effectively predict for future cases. We will use a RF, XGBoost, and Logistic Regression model to look for the best prediction model. Then we will look at our important features and do a statistical logistic regression to better understand the relationship of the features that are important with the target feature. The target feature will be the receipt of blood or not. We want to predict this to better prepare for instances of when we will need it and to better predict the need of blood products. This will help with the blood bank and knowing when we may need to start calling for more blood early.

df7.head(10)

# Lets look at some general and descriptive stats.

# Lets look at what what the length of stay may be for a patient given blood. We can also look at the icu total amount of days of someone given blood and he hospital arrival dates for those given blood to see things over time.
plt.figure(figsize=(6, 12))
plt.subplot(131)
plt.bar(df7.BLDPGVN, df7.ISS)
plt.subplot(132)
plt.bar(df7.BLDPGVN, df7.TMAT)
plt.subplot(133)
plt.plot(df7.HOSP_ARRIV_DATE, df7.BLDPGVN)

# Lets also look at the counts for the target feature alone.
plt.hist(df7.BLDPGVN)

# It looks like we have confirmed what we may have suspected. There are higher ISS scores for patients given blood. Interestingly the amount of blood given over the years has decreased. We also do not need to scale our data because we are using ensemble methods with RF and XGB as well as logistic regression which also does not require it.

# We will now sort our data by the date, we will use the date of arrival feature and then split out the sets.
df7 = df7.sort_values(by = 'HOSP_ARRIV_DATE')

# Lets see the sorted data.
print(df7['HOSP_ARRIV_DATE'])

# Next lets go ahead and get the month value out of the data time feature to create a new feature with just month. So, we can later drop the hospital arrival date after sorting.
df7['Hosp_Arr_Month'] = df7['HOSP_ARRIV_DATE'].dt.month
df7['Hosp_Arr_Month'] = df7['Hosp_Arr_Month'].astype('uint8')
df7.info()
df7['Hosp_Arr_Month'].head(10)

# Now that we have the data sorted and the month column created we will split it into a train-val-test set. We will see the what the 60-20-20 split is.

# Create the 60% split.
split_date = pd.datetime(2014,1,28)
train = df7.loc[df7['HOSP_ARRIV_DATE'] <= split_date]
train.info()

# Create the 20% split.
split_date2 = pd.datetime(2015,5,1)
val = df7.loc[(df7['HOSP_ARRIV_DATE'] > split_date) & (df7['HOSP_ARRIV_DATE'] < split_date2)]
val.info()
print(val['HOSP_ARRIV_DATE'])

test = df7.loc[df7['HOSP_ARRIV_DATE'] > split_date2]
test.info()
print(test['HOSP_ARRIV_DATE'])

# Now we will drop the datetime variable so it does not interfere with the supervised learning models and we can look at a more static prediction.
train = train.drop('HOSP_ARRIV_DATE',axis=1)
val = val.drop('HOSP_ARRIV_DATE',axis=1)
test = test.drop('HOSP_ARRIV_DATE',axis=1)

train.info()
val.info()
test.info()

# Next lets strip the datetime and convert the column to ordinal to keep the time element in tact.
#train['HOSP_ARRIV_DATE'] = train['HOSP_ARRIV_DATE'].apply(lambda x: x.toordinal())
#train.info()
#print(train['HOSP_ARRIV_DATE'])

#val['HOSP_ARRIV_DATE'] = val['HOSP_ARRIV_DATE'].apply(lambda x: x.toordinal())
#val.info()
#print(val['HOSP_ARRIV_DATE'])

#test['HOSP_ARRIV_DATE'] = test['HOSP_ARRIV_DATE'].apply(lambda x: x.toordinal())
#test.info()
#print(test['HOSP_ARRIV_DATE'])

# Create X and y for the train, val, and test set. we will also turn our data into numpy arrays to better process them in the algorithms.
y_train = train['BLDPGVN'].values
x_train = train.drop('BLDPGVN', axis = 1).values
y_val = val['BLDPGVN'].values
x_val = val.drop('BLDPGVN', axis = 1).values
y_test = test['BLDPGVN'].values
x_test = test.drop('BLDPGVN', axis = 1).values

# We will also make a set to keep as a pandas frame to use later for analysis.
y1_train = train['BLDPGVN']
x1_train = train.drop('BLDPGVN', axis = 1)
y1_val = val['BLDPGVN'].values
x1_val = val.drop('BLDPGVN', axis = 1)
y1_test = test['BLDPGVN'].values
x1_test = test.drop('BLDPGVN', axis = 1)

# Now lets make sure they are all the right size before moving on.
print(y_train.shape, x_train.shape, y_val.shape, x_val.shape, y_test.shape, x_test.shape)

print(y1_train.shape, x1_train.shape, y1_val.shape, x1_val.shape, y1_test.shape, x1_test.shape)

# Lets also see what kind of distributions of the target feature we get per set.
plt.figure(figsize=(6, 12))
plt.subplot(131)
plt.hist(train.BLDPGVN)
plt.subplot(132)
plt.hist(val.BLDPGVN)
plt.subplot(133)
plt.hist(test.BLDPGVN)

print(train['BLDPGVN'].value_counts())
print(val['BLDPGVN'].value_counts())
print(test['BLDPGVN'].value_counts())

# Now that we have our sets we can begin to model. Lets start with LogR and then we will do RF and end with XGB models. We will use Parfit to get the best parameters from the val set after we get baseline numbers.

# Logistic Regression Baseline Model.
lr = LogisticRegression()
lr.fit(x_train, y_train)

#lr4 = LogisticRegression()
#lr4.fit(x1_train, y1_train)

# Lets get initial predictions for the train and validations sets.
# The predicted probabilities for predicting the class and getting our AUC and f1 score.
LRpredprob_train = lr.predict_proba(x_train)[:, 1]
LRpredprob_val = lr.predict_proba(x_val)[:, 1]

#LR4predprob_train = lr4.predict_proba(x1_train)[:, 1]
#LR4predprob_val = lr4.predict_proba(x1_val)[:, 1]
# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
LRpreds_train = lr.predict(x_train)
LRpreds_val = lr.predict(x_val)

#LR4preds_train = lr4.predict(x1_train)
#LR4preds_val = lr4.predict(x1_val)

# Results.
print ('F1 Score',f1_score(y_train,LRpreds_train))
print ('F1 Score',f1_score(y_val,LRpreds_val))
print ('ROC AUC Score',roc_auc_score(y_train,LRpredprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,LRpredprob_val))

#print ('F1 Score',f1_score(y1_train,LR4preds_train))
#print ('F1 Score',f1_score(y1_val,LR4preds_val))
#print ('ROC AUC Score',roc_auc_score(y1_train,LR4predprob_train))
#print ('ROC AUC Score',roc_auc_score(y1_val,LR4predprob_val))

# We see we get better results on the validation set not due to underfitting, which is rare, but due to the small sample size. This makes sense but means we are still performing decently given the circumstances. But as we can see we have lower numbers, as happens with LRs, so lets tune and see if we can improve the validation numbers and the training scores compared to validation scores. We will be optimizing with parfit and not CV because we have a time element in the data and we do not want the shuffling of the samples for testing to skew the results. Also parfit allows us to optimize on the val set and not the training set like we would with CV.
grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(grid)

best_model, best_score, all_models, all_scores = bestFit(LogisticRegression(), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Now that we have found our best regularization numbers for the LR we will implement them and see how much things change.
lr2 = LogisticRegression(C = 1000, penalty = 'l2', solver = 'lbfgs', n_jobs=-1)
lr2.fit(x_train, y_train)

#lr5 = LogisticRegression(C = 100, penalty = 'l2', solver = 'newton-cg')
#lr5.fit(x1_train, y1_train)

# Again we will get the predicted probabilities for predicting the class and getting our AUC score.
LR2predprob_train = lr2.predict_proba(x_train)[:, 1]
LR2predprob_val = lr2.predict_proba(x_val)[:, 1]

#LR5predprob_train = lr5.predict_proba(x1_train)[:, 1]
#LR5predprob_val = lr5.predict_proba(x1_val)[:, 1]

# Then the decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
LR2preds_train = lr2.predict(x_train)
LR2preds_val = lr2.predict(x_val)

#LR5preds_train = lr5.predict(x1_train)
#LR5preds_val = lr5.predict(x1_val)

# The the best results for our best LR model after regularization.
print ('F1 Score',f1_score(y_train,LR2preds_train))
print ('F1 Score',f1_score(y_val,LR2preds_val))
print ('ROC AUC Score',roc_auc_score(y_train,LR2predprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,LR2predprob_val))

#print ('F1 Score',f1_score(y1_train,LR5preds_train))
#print ('F1 Score',f1_score(y1_val,LR5preds_val))
#print ('ROC AUC Score',roc_auc_score(y1_train,LR5predprob_train))
#print ('ROC AUC Score',roc_auc_score(y1_val,LR5predprob_val))

# As we can see this gets us the a really good score. However, we know we can improve the 79.07% f1 score an the 72.91% AUC score for the data. So lets look at our confusion matric and classification reports to help us figure out what we can improve moving forward.
print ('Training Confusion Matrix',confusion_matrix(y_train,LR2preds_train))
print ('Val Confusion Matrix',confusion_matrix(y_val,LR2preds_val))
print ('Training Classification report',classification_report(y_train,LR2preds_train))
print ('Val Classification Report',classification_report(y_val,LR2preds_val))

#print ('Training Confusion Matrix',confusion_matrix(y1_train,LR5preds_train))
#print ('Val Confusion Matrix',confusion_matrix(y1_val,LR5preds_val))
#print ('Training Classification report',classification_report(y1_train,LR5preds_train))
#print ('Val Classification Report',classification_report(y1_val,LR5preds_val))

# It looks like we got things going well on the first try. Lets keep moving forward with the other models just to see how much more we can improve out scores.

# We will again start with a base model before tuning it.
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

#rfc4 = RandomForestClassifier()
#rfc4.fit(x1_train, y1_train)

# Lets get initial predictions for the train and validations sets.
# The predicted probabilities for predicting the class and getting our AUC and f1 score.
RFpredprob_train = rfc.predict_proba(x_train)[:, 1]
RFpredprob_val = rfc.predict_proba(x_val)[:, 1]

#RF4predprob_train = rfc4.predict_proba(x1_train)[:, 1]
#RF4predprob_val = rfc4.predict_proba(x1_val)[:, 1]

# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
RFpreds_train = rfc.predict(x_train)
RFpreds_val = rfc.predict(x_val)

#RF4preds_train = rfc4.predict(x1_train)
#RF4preds_val = rfc4.predict(x1_val)

# Results.
print ('F1 Score',f1_score(y_train,RFpreds_train))
print ('F1 Score',f1_score(y_val,RFpreds_val))
print ('ROC AUC Score',roc_auc_score(y_train,RFpredprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,RFpredprob_val))

#print ('F1 Score',f1_score(y1_train,RF4preds_train))
#print ('F1 Score',f1_score(y1_val,RF4preds_val))
#print ('ROC AUC Score',roc_auc_score(y1_train,RF4predprob_train))
#print ('ROC AUC Score',roc_auc_score(y1_val,RF4predprob_val))

# As we can see we got an amazing score on the baseline trainig set and the validation set had a better score than our LR. But we can do better. So lets look at our confusion matric and classification reports to help us figure out what we need to improve.
print ('Training Confusion Matrix',confusion_matrix(y_train,RFpreds_train))
print ('Val Confusion Matrix',confusion_matrix(y_val,RFpreds_val))
print ('Training Classification report',classification_report(y_train,RFpreds_train))
print ('Val Classification Report',classification_report(y_val,RFpreds_val))

#print ('Training Confusion Matrix',confusion_matrix(y1_train,RF4preds_train))
#print ('Val Confusion Matrix',confusion_matrix(y1_val,RF4preds_val))
#print ('Training Classification report',classification_report(y1_train,RF4preds_train))
#print ('Val Classification Report',classification_report(y1_val,RF4preds_val))

# Lets next tune our parameters of the RF to improve our scores. We will start with the n_estimators.
Rgrid = {
    'n_estimators': [100, 300, 500, 700, 900, 1100, 1500, 2000, 2500, 3000],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(Rgrid)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next we will look at the features.
Rgrid2 = {
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(Rgrid2)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(n_estimators=500), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next we will look at the max depth.
Rgrid3 = {
    'max_depth': [1, 2, 3, 4, 5, 9, 10, 20, 45, 60, 100, 150, 200, 250, 300, 450, 550, 600],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(Rgrid3)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(n_estimators=500, max_features='auto'), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next lest tune the max leaf nodes.
Rgrid4 = {
    'max_leaf_nodes': [None, 2, 3, 5, 7, 9, 10, 15, 20, 45, 60, 65, 70, 75, 80],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(Rgrid4)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(n_estimators=500, max_features='auto',
                                                                                max_depth=10), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Finally, we will look at the min sample leaves to tune the last of the most important parameters for improvement in and RF.
Rgrid5 = {
    'min_samples_leaf': [1, 2, 3, 5, 7, 9, 10, 15, 20, 45, 60, 65, 70, 75, 80],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(Rgrid5)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(n_estimators=500, max_features='auto',
                                                                                max_depth=10, max_leaf_nodes=None), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# We now finally get to our best tuned model on the val set and will plug it in to see how much improvement we can get.
rfc2 = RandomForestClassifier(n_estimators=500, max_features='auto', max_depth=10, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, random_state=4209)

rfc2.fit(x_train,y_train)

# Lets again get the predictions for the train and validations sets. The predicted probabilities for predicting the class and getting our AUC and f1 score.
RF2predprob_train = rfc2.predict_proba(x_train)[:, 1]
RF2predprob_val = rfc2.predict_proba(x_val)[:, 1]

# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
RF2preds_train = rfc2.predict(x_train)
RF2preds_val = rfc2.predict(x_val)

# Tuned results.
print ('F1 Score',f1_score(y_train,RF2preds_train))
print ('F1 Score',f1_score(y_val,RF2preds_val))
print ('ROC AUC Score',roc_auc_score(y_train,RF2predprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,RF2predprob_val))

# As we can see we got an improved score on the tuned datasets and the validation set had a better f1 score than both our first RF and both LRs. But our AUC score did go down some. Next we will be comparing the XGBoost to these models to see if it does better. So lets look at our confusion matrix here tounderstand the  classification reports to help us figure out what we still need to improve.
print ('Training Confusion Matrix',confusion_matrix(y_train,RF2preds_train))
print ('Val Confusion Matrix',confusion_matrix(y_val,RF2preds_val))
print ('Training Classification report',classification_report(y_train,RF2preds_train))
print ('Val Classification Report',classification_report(y_val,RF2preds_val))

# Now that we have improved the model with the RF, we can next see if an XGBoost will get us any better numbers and then we can choose the best model. Lets start with getting the base model.
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

# Lets get the base predictions for the train and validations sets. The predicted probabilities for predicting the class and getting our AUC and f1 score.
xgbpredprob_train = xgb.predict_proba(x_train)[:, 1]
xgbpredprob_val = xgb.predict_proba(x_val)[:, 1]

# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
xgbpreds_train = xgb.predict(x_train)
xgbpreds_val = xgb.predict(x_val)

# Lets look at the error to assess the fit and efficacy. We will use the aucpr eval metric to get the f1 score related score. We can also use auc but we are focusing mroe on f1 score for prediction and wewill get the auc later. This is us basically re-running the fit and evaluating it vs the val set to see what we get without any tuning. But we will look at the results from above without the evaluation step to get the general baseline next.
eval_set = [(x_val, y_val)]
eval_metric = ["aucpr","error"]
%time xgb.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=2)

# We get a good base score with a low error to start lets see if we can improve. (aucpr 0.913701, error 0.125)

# Results from the initial basline model we want to improve without evaluation.
print ('F1 Score',f1_score(y_train,xgbpreds_train))
print ('F1 Score',f1_score(y_val,xgbpreds_val))
print ('ROC AUC Score',roc_auc_score(y_train,xgbpredprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,xgbpredprob_val))

# Lets look at our confusion matrix here to understand the  classification reports to help us figure out what we still need to improve.
print ('Training Confusion Matrix',confusion_matrix(y_train,xgbpreds_train))
print ('Val Confusion Matrix',confusion_matrix(y_val,xgbpreds_val))
print ('Training Classification report',classification_report(y_train,xgbpreds_train))
print ('Val Classification Report',classification_report(y_val,xgbpreds_val))

# Next lets start the tunning process.
XGgrid = {
    'learning_rate': [0.005, 0.01, 0.02, 0.1, 0.2, 0.3],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(XGgrid)

best_model, best_score, all_models, all_scores = bestFit(XGBClassifier(), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next lets tune the estimators.
XGgrid2 = {
    'n_estimators': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(XGgrid2)

best_model, best_score, all_models, all_scores = bestFit(XGBClassifier(learning_rate=0.3), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next we will be tuning the max depth.
XGgrid3 = {
    'max_depth': [1, 2, 3, 7, 9, 15, 20, 30, 40],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(XGgrid3)

best_model, best_score, all_models, all_scores = bestFit(XGBClassifier(learning_rate=0.3, n_estimators=1700), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# Next we will be tuning the max depth and the estimators together to make sure we get the same numbers as we did when they were separate. This is due to these parameters being highly dependent upon each other.
XGgrid4 = {
    'max_depth': [1, 2, 3, 7, 9, 15, 20, 30, 40],
    'n_estimators': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(XGgrid4)

best_model, best_score, all_models, all_scores = bestFit(XGBClassifier(learning_rate=0.3), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# As we can see a 2 depth and 700 estimators are now our best parameters and we will change to accomodate this. Next we will tune the colsample by tree.
XGgrid5 = {
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
    'n_jobs': [-1],
    'random_state': [4218]
}

paramGrid = ParameterGrid(XGgrid5)

best_model, best_score, all_models, all_scores = bestFit(XGBClassifier(learning_rate=0.3, n_estimators=700, max_depth=2), paramGrid,
                                                    x_train, y_train, x_val, y_val,
                                                    metric=f1_score, greater_is_better=True,
                                                    scoreLabel='F1_Score')

print(best_model, best_score)

# We will stop tunning there since those are the more important parameters. We will now plug those in and look at what improvements we made to the base model.
xgb2 = XGBClassifier(learning_rate=0.3, n_estimators=700, max_depth=2, n_jobs=-1, colsample_bytree=0.1, random_state=4218)

# Lets look at the eror to compare with the baseline and then we will update the model to then get predictions.
eval_set = [(x_val, y_val)]
eval_metric = ["aucpr","error"]
%time xgb2.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=2)

# It lookslike we will keep our parameters. The error was lower with our tuning and we can move forward with predictions.
xgb2.fit(x_train, y_train)

# Lets get the tuned predictions for the train and validations sets. The predicted probabilities for predicting the class and getting our AUC and f1 score.
xgb2predprob_train = xgb2.predict_proba(x_train)[:, 1]
xgb2predprob_val = xgb2.predict_proba(x_val)[:, 1]

# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
xgb2preds_train = xgb2.predict(x_train)
xgb2preds_val = xgb2.predict(x_val)

# Results from the tuned model.
print ('F1 Score',f1_score(y_train, xgb2preds_train))
print ('F1 Score',f1_score(y_val, xgb2preds_val))
print ('ROC AUC Score',roc_auc_score(y_train,xgb2predprob_train))
print ('ROC AUC Score',roc_auc_score(y_val,xgb2predprob_val))

# Lets look at our confusion matrix here to understand the  classification reports to help us figure out what we may have missed.
print ('Training Confusion Matrix',confusion_matrix(y_train, xgb2preds_train))
print ('Val Confusion Matrix',confusion_matrix(y_val, xgb2preds_val))
print ('Training Classification report',classification_report(y_train, xgb2preds_train))
print ('Val Classification Report',classification_report(y_val, xgb2preds_val))

# Overall, we saw that the validation set tuned f1 score for the models was: LR = 79.07% vs. RF = 91.56% vs. XGB = 93.67% (when using the eval method otherwise it was 89.4364% with an error of 0.0625). The AUC score for how the model perform on the validation set was: LR = 78.43% vs. RF = 92.92% vs. XGB = 91.56%. Overall, the XGBoost model performed the best. It had the higher f1 score even though it had a lower AUC score than RF.

# Next lets make sure that this performance was not a fluke by running our XGB on the test set. This will help us see how it does on new data that it has not seen before.

# Lets get the tuned predictions for the test data. The predicted probabilities for predicting the class and getting our AUC and f1 score.
xgb2predprob_test = xgb2.predict_proba(x_test)[:, 1]

# The decision predictions to help us classify and get the f1 scores and see what the recall and precision are if we want them.
xgb2preds_test = xgb2.predict(x_test)

# Results from the tuned model on the test data.
print ('F1 Score',f1_score(y_test, xgb2preds_test))
print ('ROC AUC Score',roc_auc_score(y_test, xgb2predprob_test))

# We can now see that there may be some other parameters we could tune due to the f1 and AUC scores being lower than the val and train data. However, it is not horrible, we do get an f1 score of 84% and an AUC score of 74.6%. The AUC score is not great so we will look at our confusion matrix and classification reports to see what we could improve on later, before moving on. But this f1 and AUC score is still pretty good for real world instances.

# Note: The XGB did still perform better than the RFC on the test data. Out of curiousity we looked at the rfc2 (tuned model) against the train data. We found that the f1 score was good at 85.98%, but the AUC was worse at 64.15%. So, we would still want the XGB to do our prediction on new data moving forward.
print ('Test Confusion Matrix',confusion_matrix(y_test, xgb2preds_test))
print ('Test Classification report',classification_report(y_test, xgb2preds_test))

# We will move forward with this model to get the feature importance, since it was still the best model, and then do the regression to see what the relationships are.
f = 'total_gain'
modelimp = xgb2.get_booster().get_score(importance_type = f)
modelimp

# Lets visualize that.
plot_importance(xgb2, importance_type = 'total_gain')
fig = plt.gcf()
fig.set_size_inches(25,40)
fig.savefig('XGB_Model_Feature_Importance', dpi=200)
plt.show()

# We can now see that features 55 the hospital arrival month, 2 weight, 3 transport agency, 0 age, 4 the ISS score, 1 height, 10 if the patient was intubated or not, 12 if a FAST was completed and positive, 26 if the batient had a high SBP, and 29 if the patient had a high HR, are our top 10 most influential based on the FIA. We will move forward with these features and identify them as well.
x1_train.info()

# Next we will start on the logistic regression with stats models to interpret the magnitudes of the relationships to know where to focus protocols and policies to improve patient outcomes. To to this we will enter the top 10 most important features from our list.

# First we will have to change the features that are in our top 10 so we can create a model.
df7['AGE'] = np.round(df7['AGE']).astype('uint8')
df7['WEIGHT'] = np.round(df7['WEIGHT']).astype('uint32')
df7['HEIGHT'] = np.round(df7['HEIGHT']).astype('uint32')
df7['INTN'] = np.round(df7['INTN']).astype('uint8')


a = df7['BLDPGVN']
b = df7.drop('BLDPGVN', axis=1)

# Lets take the top 10 features and place them in another dataset for evaluation.
df8 = b.iloc[:,[0,1,2,3,5,11,13,27,30,56]]
df8.info()
df9 = df8.values
df9a = a.values
# Now lets look at the base logistic regression model.
endo = df9a
exo = sm.add_constant(df9)

print(exo)
print(exo.shape)

LRFRF = sm.Logit(endo, exo[:,0])
resultsXG = LRFXG.fit()

# Now lets take at the base model results.
print (resultsXG.summary2())

# Now lets look at the full model to see what changes.
df8.info()
LRFXG2 = sm.Logit(endo, exo)
resultsXG2 = LRFXG2.fit()

# Now lets take at the full model results.
print (resultsXG2.summary2())

# This is very interesting, we see that only 2 of our models values are significant. Lets see if the model improves any by taking out the highest non-significant values in a backward order.
df10 = df8.drop('ED_HR2_High',axis=1)
exo2 = sm.add_constant(df10)
df10.info()
LRFXG3 = sm.Logit(endo, exo2)
resultsXG3 = LRFXG3.fit()
print (resultsXG3.summary2())

# Lets now take out the next highest p value. We see that our AIC and BIC decreased. We will try to determine if continuing to take our these non-significant values in order helps us any.
df11 = df10.drop('WEIGHT',axis=1)
exo3 = sm.add_constant(df11)
df11.info()
LRFXG4 = sm.Logit(endo, exo3)
resultsXG4 = LRFXG4.fit()
print (resultsXG4.summary2())

# Lets now take out the next highest p value. We see that our AIC and BIC decreased. We will try to determine if continuing to take our these non-significant values in order helps us any.
df12 = df11.drop('FASTCOMPOS',axis=1)
exo4 = sm.add_constant(df12)
df12.info()
LRFXG5 = sm.Logit(endo, exo4)
resultsXG5 = LRFXG5.fit()
print (resultsXG5.summary2())

# Lets now take out the next highest p value. Because we keep seeing that our AIC and BIC decrease. We will continue to try to determine if continuing to take our these non-significant values in order helps us any.
df13 = df12.drop('INTN',axis=1)
exo5 = sm.add_constant(df13)
df13.info()
LRFXG6 = sm.Logit(endo, exo5)
resultsXG6 = LRFXG6.fit()
print (resultsXG6.summary2())

# Again lets take out the next highest p value. We saw another great decrease in our AIC and BIC. We will again try to determine if continuing to take our these non-significant values in order helps us any.
df14 = df13.drop('HEIGHT',axis=1)
exo6 = sm.add_constant(df14)
df14.info()
LRFXG7 = sm.Logit(endo, exo6)
resultsXG7 = LRFXG7.fit()
print (resultsXG7.summary2())

# Again lets take out the next highest p value, this may be the last time since things are starting to get really good. We saw another great decrease in our AIC and BIC. We will again try to determine if continuing to take our these non-significant values in order helps us any.
df15 = df14.drop('Hosp_Arr_Month',axis=1)
exo7 = sm.add_constant(df15)
df15.info()
LRFXG8 = sm.Logit(endo, exo7)
resultsXG8 = LRFXG8.fit()
print (resultsXG8.summary2())

# Now we will look at if ISS should come out or not. That would leave us with 3 features that should be highly considered from the top 10 what predicting who will receive blood or not. So, again lets take out the next highest p value, and we know this will be the last time since things are now very good with the model. We saw another great decrease in our AIC and BIC. We will again try to determine if continuing to take our these non-significant values in order helps us any.
df16 = df15.drop('ISS',axis=1)
exo8 = sm.add_constant(df16)
df16.info()
LRFXG9 = sm.Logit(endo, exo8)
resultsXG9 = LRFXG9.fit()
print (resultsXG9.summary2())

# We can now see that we have a very significant model. This means that there 3 items should be the top focus of our efforts in protocol, policy, and management when predicting who will need blood in future encounters. We will focus on the BIC number because a false positive (a patient thought to need blood) could be as or more misleading as a patient that does not. This would cause a waste of resources that are finite (the blood from the blood bank), if the patient does not need the blood.

# We will now get some odds ratios to make this more digestable for all audiences and to helps us with a presentation. We will first get the odds to the significant model, then we will look at the odds of the full model to still get an idea of what relationships we are looking at when all 10 are considered.
model_odds = pd.DataFrame(np.exp(resultsXG9.params), columns= ['OR'])
model_odds['p-value']= resultsXG9.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(resultsXG9.conf_int())
model_odds

model_oddsfull = pd.DataFrame(np.exp(resultsXG2.params), columns= ['OR'])
model_oddsfull['p-value']= resultsXG2.pvalues
model_oddsfull[['2.5%', '97.5%']] = np.exp(resultsXG2.conf_int())
model_oddsfull

# Overall, there does seem to be a significance in the age, transport agency, and if a patients blood pressure is high having a significant correclation with the need for blood. We can see that patients that have a higher blood pressure have 51% less likely chance of needing blood products. We can see a similar outcom with the transport agency. Depending on the transport agency we will see a 5% less likely chance of the need for blood. This tells us that we may need to split out the agencies and see which ones may have protocols, procedures, personnel, operations, or even resources that contribute or detract from a patient needing blood products.

# Finally, we may also need to consider that patients who arrive with higher blood pressures tend to not be decompensating yet. This means they are in better shape than patients who arrive with a lower blood pressure. Finally, we see that patients age makes them 6% more likely to receive blood products. We would need to also in the future look at what ages receive blood products more by splitting out the age groups or individual ages. Other honorable mentions from the full model were that when the patients received a fast and it was positive they were 30% more likely to receive blood products. Also, when a FAST that was positive was present in the model patients with higher systolic BPs were 29% more likely to receive blood. In the full model age accounted for patients being 14% more likely to receive blood, while a high heart rate and the month that the patient arrives saw patients 1% and 6% more likely to receive blood. Hopefully we can gain further information from this study and have future directions for investigations in the future.

#next we will be looking to export our dataset to excel in order to perform unsupervised learning analysis on it.
df7.to_excel (r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python\hermann_df7.xlsx', header=True)

df8.to_excel (r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python\hermann_df8.xlsx', header=True)

train.to_excel (r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python\hermann_train.xlsx', header=True)

val.to_excel (r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python\hermann_val.xlsx', header=True)

test.to_excel (r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python\hermann_test.xlsx', header=True)