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
data_path = "static/CC_general.csv"
data = pd.read_csv(data_path)
st.write(data.head())

# Handling missing values (if any)
data.fillna(data.mode(), inplace=True)
data.fillna(data["CREDIT_LIMIT"].mode()[0], inplace=True)

# Selecting relevant features for clustering
features = data[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CREDIT_LIMIT', 'PAYMENTS',
       ]]
features.dropna()

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

st.write(f"NaN values: {np.isnan(scaled_features).sum()}")
st.write(f"Infinity values: {np.isinf(scaled_features).sum()}")
