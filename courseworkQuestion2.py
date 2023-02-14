# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:53:51 2021

@author: 44778
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans
import sklearn.cluster as cluster

df = pd.read_csv('wholesale_customers.csv')

df.drop(columns=['Channel','Region'], inplace=True)

print(df)

# 2.1

df_mean = []
df_min = []
df_max = []

for col in df:
    df_mean.append(df[col].mean())
    df_min.append(df[col].min())
    df_max.append(df[col].max())

# 2.2

numpyDf = df.to_numpy()

kmeans = kmeans(n_clusters=3).fit(numpyDf)
centers = kmeans.cluster_centers_
label = kmeans.fit_predict(numpyDf)



filtered_label0 = numpyDf[label == 0]

plt.figure()
# keep a reference to the first axis

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[0])
ax1.set_ylabel(df.columns[1])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 1],c=label)

# and a reference to the second axis
ax2.set_xlabel(df.columns[0])
ax2.set_ylabel(df.columns[2])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 2],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[0])
ax1.set_ylabel(df.columns[3])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[0])
ax2.set_ylabel(df.columns[4])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[0])
ax1.set_ylabel(df.columns[5])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[1])
ax2.set_ylabel(df.columns[2])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[1])
ax1.set_ylabel(df.columns[3])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[1])
ax2.set_ylabel(df.columns[4])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[1])
ax1.set_ylabel(df.columns[5])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[2])
ax2.set_ylabel(df.columns[3])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[2])
ax1.set_ylabel(df.columns[4])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[2])
ax2.set_ylabel(df.columns[5])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel(df.columns[3])
ax1.set_ylabel(df.columns[4])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

ax2.set_xlabel(df.columns[3])
ax2.set_ylabel(df.columns[5])
ax2.scatter(numpyDf[:, 0], numpyDf[:, 4],c=label)

plt.tight_layout()
plt.show()

fig, (ax1) = plt.subplots(1, 1)

ax1.set_xlabel(df.columns[4])
ax1.set_ylabel(df.columns[5])
ax1.scatter(numpyDf[:, 0], numpyDf[:, 3],c=label)

plt.tight_layout()
plt.show()

# 2.3

M = len(numpyDf)
BC = np.zeros( 11 ) # between cluster
WC = np.zeros( 11 ) # within cluster
kSet = [3,5,10]

for K in kSet:
    km = cluster.KMeans( n_clusters=K )
    km.fit(numpyDf)
    

    members = [[] for i in range( K )] 
    for j in range( M ): 
        members[ km.labels_[j] ].append( j )
    
    within = np.zeros(( K ))
    for i in range( K ): # loop through all clusters
        within[i] = 0.0
        for j in members[i]:
            within[i] += ( np.square( numpyDf[j,0]-km.cluster_centers_[i][0] ) + np.square( numpyDf[j,1]-km.cluster_centers_[i][1] ))
    WC[K] = np.sum( within )

    between = np.zeros(( K ))
    for i in range( K ): 
        between[i] = 0.0
        for l in range( i+1, K ): 
            between[i] += ( np.square( km.cluster_centers_[i][0]-km.cluster_centers_[l][0] ) + np.square( km.cluster_centers_[i][1]-km.cluster_centers_[l][1] ))
    BC[K] = np.sum( between )
    
    #-compute overall clustering score
    score = BC[K] / WC[K]
    
    print('K={}  WC={}  BC={}  score={} '.format( K, WC[K], BC[K], score))












