# -*- coding: utf-8 -*-
"""
@author: Dr. acdaviano
"""

# Lets import all of our packages to do our analysis.
import numpy as np
import sys
import modin.pandas as pd
from kmodes.kmodes import KModes
import scipy as spy
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
from matplotlib.pyplot import figure
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from functools import partial
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
import os
import sklearn.cluster as cluster
import time
import matplotlib.colors as mcolors

%matplotlib inline

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

#Change my working directory to more easily locate my data files.
os.chdir(r'C:\Users\acdav\OneDrive\Documentos\OneDrive\Alexjandro\research\Python')

#View my working directory.
os.getcwd()

#Import the data from Memorial Hermann Hospital that we have already cleaned and explored in the 'revisit hermann' supervised analysis. This allows us to get into the unsupervised learning analysis.
df = pd.read_excel(r'hermann_df7.xlsx')

#complete some initial visualization of the revised data.
df.info()
columns = df.columns
columns
df.head()

#create a copy of the data to be able to work with it so we dont overwrite it. We will also drop our datatime object since we have the month already and the data is already in order from date. We will also drop the ID number created from python when we exported the dataset.
df1 = df.drop(['HOSP_ARRIV_DATE','Unnamed: 0'],axis=1)

# Lets also change the dataframe size to use less memory.
df1 = df1.astype('uint8', copy=True, errors='raise')
df1.info()

# Next we will scale our continuous values which are only age, height, weight, and ISS score.
concoldf = df1.iloc[:,[0,1,2,4]]
concoldf.info()
concoldf_scale = concoldf.columns

dfx = concoldf.values
stan_scaler = StandardScaler()
dfx_scale = stan_scaler.fit_transform(dfx)
concoldf_test = pd.DataFrame(dfx_scale, columns=concoldf_scale)
l_testconcoldf = concoldf.drop(concoldf_scale, axis=1)
concoldf = pd.concat([concoldf_test, l_testconcoldf], axis=1)
concoldf.info()

# Now we will add the scaled data back to the original dataframe.
concoldf_test = pd.DataFrame(dfx_scale, columns=concoldf_scale)
Newdf = df1.drop(concoldf_scale, axis=1)
df2 = pd.concat([concoldf_test, Newdf], axis=1)
df2.info()

# With this analysis we will be looking at both ground truth and unknown situations. We want to explore what contributes to the death of a patient with HCA. WE then want to look at clustering blood products being given to better explore our predictions from the supervised learning algorithms. We will also explore the thought of the month having an influence on blood being given, so we will look at what may exist there with GMM. Then finally we will look for other patterns that may exist with DBSCAN and see if we can attribute the clusters made to any new insights from the current features. We will perform all of the above with the overall data and for curiosity the significant data from the supervised model as well. Lets get started!

# First we will look to see if there is a heirachy in the data with HCA looking to cluster for a death outcome. So, we will create a dataset without the death outcome and then we will run our HCA model to see what we get.

# taking out the target to do predictive clustering.
# for the standard data.
dfhca = df2.drop('out',axis=1).values
dfhca_y = df2['out'].values

print(dfhca)

# for the unstandardized data, we do this to explore the differences since standardizing can actually hurt our patterns in some cases. We will compare the results to both the standardized and unstandardized data.
dfhca2 = df1.drop('out',axis=1).values
dfhca2_y = df1['out'].values
print(dfhca2)

# Lets take an exploratory look at age and height by the group of death vs. lived for the patients.
plt.scatter(df2.iloc[:,0], df2.iloc[:,1],c=df['out'], cmap='gist_rainbow')

# Now lets start with agglomerative HCA.
# Create the dendrograms with unstandardized first and standardized second. we will use the average method due to there being different measures in the data, as well as the manhattan method later on for the same reasoning.
dendrogram2 = sch.dendrogram(sch.linkage(dfhca2, method='ward'))

dendrogram = sch.dendrogram(sch.linkage(dfhca, method='ward'))

# We can see that after the dendrograms come out we get a more reasonable result from the unscaled data, so we will continue with the unscaled data only for prediction purposes.

# Now we will create the clusters
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

# save clusters for chart, with the standardized first and the unstandardaized second.
y_hc2 = hc.fit_predict(dfhca2)
labels2 = hc.labels_

# Lets look at the clusters that we found.
print(labels2)

unique_elements, counts_elements = np.unique(labels2, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Lets visualize the clusters from the dendrogram.
plt.scatter(dfhca2[labels2 ==0,0], dfhca2[labels2 == 0,1], s=100, c='red')
plt.scatter(dfhca2[labels2 ==1,0], dfhca2[labels2 == 1,1], s=100, c='black')
plt.show()

# Now lets compare the predicted clusters and the actual clusters.
plt.scatter(dfhca2[:,0], dfhca2[:,1],c=labels2, cmap='gist_rainbow')
plt.scatter(dfhca2[:,0], dfhca2[:,1],c=dfhca2_y, cmap='jet')

# Predicted
unique_elements, counts_elements = np.unique(labels2, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Actual
unique_elements, counts_elements = np.unique(dfhca2_y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Lets look at the silouette score for hca at this point.
silhouette_avg = silhouette_score(dfhca2, y_hc2)
silhouette_avg

# We can see that the silouette score is not great, but it is not bad either. We also can start to understand that HCA may not be the best algorithm to use. Next we will look at what Kmodes tells us since we have a majority of categorical features. Then, if we can, we will use DBSCAN to let us know more about the data without gorund truth.

# lets do kmodes with 2 for our k, we will used the scaled and unscaled data for this to again see which perfoms better.
kmo = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)

clusters = kmo.fit_predict(dfhca)
clusters2 = kmo.fit_predict(dfhca2)

# Print the cluster centroids
print(kmo.cluster_centroids_)

# Next we will look at the clusters/labels.
kmc = kmo.labels_
print(kmc)

kmc_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
print(kmc_clusters_)

kmc2_clusters_ = len(set(clusters2)) - (1 if -1 in clusters2 else 0)
print(kmc2_clusters_)

# Great so we see that we get 2 clusters again like with our HCA. Now lets plot them and get a silouette score to see if we did any better than HCA.
# Now lets compare the predicted clusters and the actual clusters. The scaled data will be first, then the unscald, and then the actual data.
plt.scatter(dfhca2[:, 0], dfhca2[:, 1],c=clusters, cmap='gist_rainbow')
plt.scatter(dfhca2[:, 0], dfhca2[:, 1],c=clusters2, cmap='gist_rainbow')
plt.scatter(dfhca2[:, 0], dfhca2[:, 1],c=dfhca2_y, cmap='jet')

# Predicted for the scaled data
unique_elements, counts_elements = np.unique(clusters, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Predicted for the unscaled data
unique_elements, counts_elements = np.unique(clusters2, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Actual
unique_elements, counts_elements = np.unique(dfhca2_y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# As we can see the scaled data does better than the unscaled data with partitioning. However, this does not mean that our model is better just because the number of observations in the predicted model are closer to the actual. We could have completely wrong values in th results. So, lets see if we did do better or if the results are worse. We will get the silouette score next to see which is true.
silhouette_avg = silhouette_score(dfhca, clusters)
silhouette_avg

silhouette_avg = silhouette_score(dfhca, clusters2)
silhouette_avg

# As we can see we actually did worse than the HCA. So, in this case it seems the HCA is better in a scenario to try to predict and maybe even get some deeper insight into the data. We will next move to more of an exploratory mindset and see what we can find out about the data through HDBSCAN and HCA again including all of the features in the model and not leaving out the deaths.

# First we will start with HDBSCAN, this is a great algorithm to help us find clusters and make sense of data when there are varying densities in the data such as real world data usually presents us. We know we do not have a lot of data so this is another good way to see what kind of patters may be present while being able to use smaller cluster sizes and amounts of data that can be allocated to those clusters. We will make our min cluster size around 20 since that would be enough to be able to draw some possible conclusions from in a data set of just over 300. We will also set the min sample to 1 because we want to be aggressive and not conservative in finding all the necessary clusters.
# Predicted for the data.
df3 = df2.values

dbc = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=1)
clustersHD = dbc.fit_predict(df3)
labelsHD = dbc.labels_
labelsHD

unique_elements, counts_elements = np.unique(clustersHD, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Now lets see what both sets of clusters look like.
plt.scatter(df3[:,0], df3[:,1],c=clustersHD, cmap='gist_rainbow')

# As we can see above that we had 2 clusters created and not much noise. A lot of the data points went to on of the clusters. We should explore what these clusters mean. Lets do That after looking at the HCA one more time with the out feature included in the data. We will get the dendrogram, create the model and then look at the results after getting the clusters.

dendrogram = sch.dendrogram(sch.linkage(df3, method='ward'))

# Now we will create the clusters. We will go with 3 since identified 3 clusters in the data.
hca = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# Save clusters for chart, with the standardized first and the unstandardaized second.
hc3 = hca.fit_predict(df3)
labels3 = hca.labels_

# Lets looks at the clusters that we found.
print(labels3)

unique_elements, counts_elements = np.unique(labels3, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Lets look at the clusters from the dendrogram.
plt.scatter(df3[labels3 == 0, 0], df3[labels3 == 0, 1], s=100, c='red')
plt.scatter(df3[labels3 == 1, 0], df3[labels3 == 1, 1], s=100, c='black')
plt.scatter(df3[labels3 == 2, 1], df3[labels3 == 2, 2], s=100, c='blue')
plt.show()

# Now that we see that 3 clusters may be more appropriate, lets try to figure out what they mean. We will look at the results fro the HDBSCAN as well as th HCA and then see if we come to a conclusion on what everything means before moving to summarize the project. We will look at HDBSCAN first then HCA.
df4 = df1
df4['hdbscan'] = labelsHD
df4['hca'] = labels3
df14 = df4.drop('hca', axis=1)
df24 = df4.drop('hdbscan', axis=1)
df5 = df3
df6 = np.hstack((df5, np.atleast_2d(labelsHD).T))
df6.shape
df7 = np.hstack((df6, np.atleast_2d(labels3).T))
df7.shape

# Next lets look at HCAs results, first we will be taking out the features that are not in the top 10 most important that we found in the supervised learning portion of this project. That will leave us with 10 features to look further into with the clusters and help give us some more perspective. First we will visualize the top 10 most impotant features from the supervised learning analysis.
df4.info()

# Next we will be putting that data into lists to be able to create new datasets to use for this analysis with the hca and hdbscan clusters added to each.
top10HCA = ['AGE', 'HEIGHT', 'WEIGHT', 'TRANS_AGENCY', 'ISS', 'INTN', 'FASTCOMPOS', 'ED_SBP2_High', 'ED_HR2_High', 'Hosp_Arr_Month', 'hca']
top10HD = ['AGE', 'HEIGHT', 'WEIGHT', 'TRANS_AGENCY', 'ISS', 'INTN', 'FASTCOMPOS', 'ED_SBP2_High', 'ED_HR2_High', 'Hosp_Arr_Month', 'hdbscan']

# The new datasets for each models clusters.
df34 = df4[top10HCA]
df44 = df4[top10HD]

# Lets visualize those datasets to ensure we are good to move forward.
df34.info()
df44.info()

# Now lets look at the medians an means of each of the features as they pertain to the clusters to get an idea of what they may mean. We can then try to get a logistic regression and multinomial logistic regression done to see if it adds to anything we may need to or want to know.
df_medianHCA = (df34.loc[df34.hca >= 0, :].groupby(df34.hca).median())
df_meanHCA = (df34.loc[df34.hca >= 0, :].groupby(df34.hca).mean())
pd.set_option('display.max_columns', None)
print(df_medianHCA)
print(df_meanHCA)

# After seeing that it looks like we have a pretty good idea of what our clusters may mean but lets see if we can get any insight from a correlation matrix. Lets split the clusters up to get a detailed idea about all of them separately.
df32 = pd.get_dummies(df34['hca'], prefix = "hca")
df32.info()
df72 = pd.concat([df32, df34], axis=1)
df72.info()
df82 = df72.drop('hca', axis=1)
df82.info()

# Lets take a look at a correlation matrix to help us out here since we did not have much luck with the logits for HCA.
hcacorr = df82.corr()
print(hcacorr)
fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(df82.corr(), annot=True)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

# We can see that the logit excluded the first cluster as a reference cluster to the other and did not incluse the resultsin the model. For that reason we will base out inferences on the existing cluster coeficient, but further, on the medians we took above on each feature as they pertain to each cluster to get a more full picture of what everything means.

# Next we will do the same to the hdbscan clusters and use a logistic regression after the medians as well to get a more full picture.
df_medianHD = (df44.loc[df44.hdbscan >= 0, :].groupby(df44.hdbscan).median())
df_meanHD = (df44.loc[df44.hdbscan >= 0, :].groupby(df44.hdbscan).mean())
pd.set_option('display.max_columns', None)
print(df_medianHD)
print(df_meanHD)

# Above we can see a more clear picture of what the clusters mean. But lets just get the full picture by runing the same multinomial logistic regression next. Then we will wrap things up with which model tells us what about our data. First we will drop the noise values to only keep the clusters. First we will change the values in the cluster column.
df44 = df44[df44.hdbscan != -1]
df44.info()
df54 = pd.get_dummies(df44['hdbscan'], prefix = 'hdbscan')
df54.info()
df64 = pd.concat([df44, df54], axis=1)
df64.info()
df94 = df64.drop('hdbscan', axis = 1)
df94.info()

# Lets take a look at a correlation matrix to help us get more understanding from hdbscan.
hdbscancorr = df94.corr()
print(hdbscancorr)
fig, ax = plt.subplots(figsize=(50,50))
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
sns.heatmap(df94.corr(), annot=True)

# In conclusion it looks like both models give us slightly different insight. However we can see that what we are being told is very different among the algorithms which still provide us with perspective. The HCA algorithm is showing us that three clusters of patients exist. Cluster 0 tells us that these patients aresimilar to hose in cluster 2. They are slightly younger, they are smaller in weight and height, they are usually more critical because they have higher ISS scores, are flown to the hospital by helicopter service, and are intubated. These patients are slightly more likely to have a FAST be completed and positive, and have lower heart rates and systolic blood pressures on average, and they are transported to the hospital in later summer months than those children in cluster 1. The correlations reflected similar insights, but told us a bit more with magnitude as well. Patients in cluster 2 had higher correlations with the transport agencies (meaning those who are flown are in worse shape most likely needing blood products). The age seems to also be slightly more correlated with the transport agency and type as well. it seems that higher ages, weights, and heights have a higher correlation with being transported by ground and not needing intubation than those who were of lower demographics. We also see that there is more of a correlation between those that receive blood products and the median transport agency in cluster 2 (PHI which is an agency that is not known for carrying blood products, but this was blood reciept at the hospital, so that could mean that these patients need the blood more due to severity and unavailable blood products prehospital). This makes a lot of sense because MH Life Flight the median agency of cluster 0 and AMR the median agency of ground transport have negative correlations with needing blood upon arrival. That means that AMR may have lower severity for patient injury and illness while MH Life Flight does in fact carry blood with them. So, That would lower the need for blood products upon arrival at the facility. This theory is confirmed by the correlations of the ISS scores and the completed FASTs that were positive. These correlations are higher with flight agencies with ISS being highest with MHL. PHI also had high correlations with FASTs being positive and completed but that could mean that they are more likely to receive one upon arrival, while MHL is capable of doing them in flight. There are also high correlations with high systolic blood pressure and heart rates possibly due to patients who are transported by ground being able to compensate better as well as being older and larger.

#The HDBSCAN numbers and correlation just split things up into the types of patients that are flown to the facility. This focused on more severe injury and more of an age gap with similar body types of the patients. patients for both clusters were severe due to the ISS scores being higher, intubation being present, higher pulse rates and lower systolic blood pressures being present as well (showing that patients were in decline and showing signs of decompensation). This algorithm chose to focus on "anomoly" as opposed to classification. The grand transport patients were seen as noise and the correlations with the flight patients were more to differentiate the possible care given to the patients from PHI nd MHL which were the median agencies and how we would define the clusters. Overall, we can learn a lot from both result sets. However, the HCA did find the hierarchy in the data and gives us more information about the patients that are transported overall to Memorial Hermann Hospital Houston Texas Medical Center.
