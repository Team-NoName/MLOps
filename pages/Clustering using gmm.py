# Import streamlit application
import streamlit as st

st.title("GMM Clustering")

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import f1_score

credit_card = pd.read_csv('https://raw.githubusercontent.com/Team-NoName/MLOps/main/static/CC_general.csv')
st.write("Credit Card Dataset:", credit_card.head())

credit_card.drop(["CUST_ID"], axis=1, inplace=True)
credit_card = credit_card.fillna(method='ffill')
st.write("Processed Credit Card Dataset:", credit_card.head())

normalized_data = normalize(credit_card)
st.write("Normalized Data:", normalized_data[:5])

pca = PCA()
pca.fit(normalized_data)
explained_var = np.cumsum(pca.explained_variance_ratio_)
fig = px.area(
    x=range(1, explained_var.shape[0] + 1),
    y=explained_var,
    labels={"x": "# Components", "y": "Explained Variance"}
)

st.plotly_chart(fig)

pca_data = PCA(n_components=3)
credit_pca = pd.DataFrame(pca_data.fit_transform(normalized_data), columns=['Component 1','Component 2','Component 3'])
st.write("PCA Data:", credit_pca.head())

inertia = []
for i in range(1,10):
    cluster = KMeans(n_clusters=i)
    cluster.fit(credit_pca)
    inertia.append(cluster.inertia_)
fig = px.line(inertia, title="Elbow graph for KMeans Clustering", labels={"index": "Clusters", "value": "Inertia"})
st.plotly_chart(fig)

cluster_spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
cluster_spectral.fit(credit_pca)
credit_card['Labels'] = cluster_spectral.labels_
fig = px.scatter_3d(x=credit_pca['Component 1'], y=credit_pca['Component 2'], z=credit_pca['Component 3'], color=cluster_spectral.labels_, size_max=18)
st.plotly_chart(fig)

fig = px.scatter(credit_card, x='BALANCE', y='PURCHASES', color='Labels', title='Balance vs Purchases')
st.plotly_chart(fig)

fig = px.scatter(credit_card, x='CREDIT_LIMIT', y='PAYMENTS', color='Labels', title='Credit Limit vs Payments')
st.plotly_chart(fig)

# Additional section with synthetic data, KMedoids clustering, and F1 score calculation
st.header("Synthetic Data Clustering and Classification")

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
