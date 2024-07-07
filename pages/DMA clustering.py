# Import streamlit application
import streamlit as st

st.title("MLOps")

import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/Team-NoName/MLOps/main/static/CC_general.csv')
# Removing full row if any attribute is missing.
df.dropna(inplace = True)
df.shape

df = df.iloc[:, 1:]

import pandas as pd

df.fillna(df.mean(), inplace=True)

# Check for missing values
missing_values = df.isnull().sum()

# Print missing values count for each column
st.write("Missing Values Count:")
st.write(missing_values)

# Check if there are any missing values left
if missing_values.sum() == 0:
    st.write("No missing values remain.")
else:
    st.write("There are still missing values in the DataFrame.")

from sklearn.cluster import KMeans

# Instantiate KMeans with k=5 and explicitly set n_init
kmeans = KMeans(n_clusters=5, n_init=10)  # You can set n_init to any desired value

# Fit KMeans to the data
kmeans.fit(df)

# Fit KMeans to the data
kmeans.fit(df)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels

# Print the count of data points in each cluster
st.write("Count of data points in each cluster:")
st.write(df['Cluster'].value_counts())

import matplotlib.pyplot as plt

# Choose features for plotting
feature1 = 'PURCHASES'
feature2 = 'BALANCE'

# Plot each cluster with a different color
fig = plt.figure(figsize=(10, 6))

for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {cluster}', alpha=0.7)

# Add labels and title
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('KMeans Clustering')
plt.legend()
plt.grid(True)
#plt.show()
st.pyplot(fig)

from sklearn_extra.cluster import KMedoids

# Assuming 'df' is your DataFrame with the relevant columns
# Assuming you have already handled missing values and scaled the data if necessary

# Choose the number of clusters
n_clusters = 5

# Instantiate KMedoids with the number of clusters
kmedoids = KMedoids(n_clusters=n_clusters)

# Fit KMedoids to the data
kmedoids.fit(df)

# Get the cluster labels
cluster_labels = kmedoids.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels

# Print the count of data points in each cluster
st.write("Count of data points in each cluster:")
st.write(df['Cluster'].value_counts())

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=86)

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

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Choose the number of clusters
n_clusters = 5

# Instantiate AgglomerativeClustering with the number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)

# Fit AgglomerativeClustering to the data
agg_clustering.fit(df)

# Get the cluster labels
cluster_labels = agg_clustering.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels

# Print the count of data points in each cluster
st.write("Count of data points in each cluster:")
st.write(df['Cluster'].value_counts())

# Plotting
fig2 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PURCHASES', y='BALANCE', hue='Cluster', palette='Set1')
plt.title('Hierarchical Clustering')
plt.xlabel('PURCHASES')
plt.ylabel('BALANCE')
plt.legend(title='Cluster')
plt.grid(True)
# plt.show()
st.pyplot(fig2)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=64)

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=2)
cluster_labels = clustering.fit_predict(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier using the clusters as features
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='binary')  # or 'macro', 'micro', 'weighted'

st.write(f'F1 Score: {f1}')
