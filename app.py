import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Title of the application
st.title("MLOps")

# Load and display the dataset
data_path = "https://raw.githubusercontent.com/Team-NoName/MLOps/main/static/CC_general.csv"
data = pd.read_csv(data_path)
st.write(data.head())

# Handling missing values (if any)
data.fillna(data.mode().iloc[0], inplace=True)
data.fillna(data["CREDIT_LIMIT"].mode()[0], inplace=True)

# Selecting relevant features for clustering
features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
                 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CREDIT_LIMIT', 'PAYMENTS']]
features.dropna(inplace=True)

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

st.write(f"NaN values: {np.isnan(scaled_features).sum()}")
st.write(f"Infinity values: {np.isinf(scaled_features).sum()}")

# Compute the linkage matrix
linked = linkage(scaled_features, method='ward')

# Plot the dendrogram
fig = plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
st.pyplot(fig)

# Fit the Agglomerative Clustering model
agg_clustering = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
agg_clustering.fit(scaled_features)

# Adding the cluster labels to the original dataset
data['Cluster'] = agg_clustering.labels_

# Reduce dimensions with PCA for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Plot the clusters
fig2 = plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=data['Cluster'], palette='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot(fig2)

# Compute the cluster centers (only for numeric columns)
numeric_cols = features.columns
cluster_centers = data.groupby('Cluster')[numeric_cols].mean()

# Display the cluster centers
st.write(cluster_centers)

# Set the plot size
fig3 = plt.figure(figsize=(14, 8))

# Create a heatmap for the cluster centers
sns.heatmap(cluster_centers, annot=True, cmap="viridis", linewidths=.5)
plt.title('Cluster Centers Heatmap')
plt.xlabel('Features')
plt.ylabel('Clusters')
st.pyplot(fig3)

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

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
