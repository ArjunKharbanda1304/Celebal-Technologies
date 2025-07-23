# -------------------------------
# Customer Segmentation Assignment
# -------------------------------
# Grouping customers based on purchasing behavior and demographics
# to target marketing strategies effectively
# -------------------------------

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# For ignoring warnings in output
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Step 1: Load and explore dataset
# -------------------------------

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Basic info about dataset
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# -------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# -------------------------------

# Basic statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=data, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Annual Income distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='orange')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()

# Spending Score distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True, color='green')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.show()

# Pairplot to see relationships
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.show()

# -------------------------------
# Step 3: Preprocessing
# -------------------------------

# Encode 'Gender' column (Male=1, Female=0)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features to use for clustering
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -------------------------------
# Step 4: Find optimal number of clusters
# -------------------------------

# Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette Scores for more validation
print("\nSilhouette Scores:")
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"For n_clusters = {i}, Silhouette Score = {score:.3f}")

# -------------------------------
# Step 5: Apply KMeans Clustering
# -------------------------------

# From Elbow and Silhouette, let's choose k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# View dataset with cluster labels
print("\nDataset with Cluster Labels:")
print(data.head())

# -------------------------------
# Step 6: Visualize Clusters
# -------------------------------

# 2D scatter plot: Annual Income vs Spending Score
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set1', data=data, s=100)
plt.title('Customer Segments (Clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# -------------------------------
# Step 7: Analyze & Interpret Clusters
# -------------------------------

# Average values in each cluster
cluster_summary = data.groupby('Cluster').mean().round(1)
print("\nCluster Summary (Mean values):")
print(cluster_summary)

# Count of customers in each cluster
cluster_counts = data['Cluster'].value_counts().sort_index()
print("\nNumber of Customers in each Cluster:")
print(cluster_counts)

# -------------------------------
# Insights (You can include these in report)
# -------------------------------
# Example insights for marketing:
# - Cluster 0: Young high spenders → Target with premium offers
# - Cluster 1: Middle-aged low spenders → Engage with discounts
# - Cluster 2: High income, moderate spending → Upsell luxury products
# - Cluster 3: Low income, high spending → Introduce budget-friendly loyalty programs
# - Cluster 4: Senior customers → Promote essential products and services
