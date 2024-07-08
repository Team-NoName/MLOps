# Import streamlit application
import streamlit as st

st.title("Spectural Clustering")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import SpectralClustering , KMeans
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score 

credit_card = pd.read_csv('https://raw.githubusercontent.com/Team-NoName/MLOps/main/static/CC_general.csv')
credit_card

credit_card.drop(["CUST_ID"], axis=1, inplace=True)
credit_card = credit_card.fillna(method='ffill')
credit_card

normalized_data = normalize(credit_card)
normalized_data

pca = PCA()
pca.fit(normalized_data)
explained_var = np.cumsum(pca.explained_variance_ratio_)
px.area(
    x=range(1, explained_var.shape[0] + 1),
    y=explained_var,
    labels={"x": "# Components", "y": "Explained Variance"}
)

pca_data = PCA(n_components=3)
credit_pca = pd.DataFrame(pca_data.fit_transform(normalized_data), columns=['Component 1','Component 2','Component 3'])
credit_pca

inertia = []
for i in range(1,10):
    cluster = KMeans(n_clusters=i)
    cluster.fit(credit_pca)
    inertia.append(cluster.inertia_)
px.line(inertia, title="Elbow graph for Spectral Clustering",labels={"index":"Clusters","value":"Inertia"})

cluster_spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
cluster_spectral.fit(credit_pca)
credit_card['Labels'] = cluster_spectral.labels_
px.scatter_3d(x=credit_pca['Component 1'],y=credit_pca['Component 2'],z=credit_pca['Component 3'], color=cluster_spectral.labels_,size_max=18)

px.scatter(credit_card,x='BALANCE', y='PURCHASES',color='Labels',title='Balance vs Purchases')

px.scatter(credit_card,x='CREDIT_LIMIT', y='PAYMENTS',color='Labels', title='Credit limit vs Payments')

from sklearn.metrics import f1_score

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=15, n_classes=2, random_state=89)

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=2, random_state=42)
cluster_labels = kmedoids.fit_predict(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='binary')  # or 'macro', 'micro', 'weighted'

st.write(f'F1 Score: {f1}')
