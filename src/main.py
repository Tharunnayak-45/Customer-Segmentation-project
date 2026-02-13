import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
print("Libraries Imported Successfully\n")

# 2. Create outputs folder (if not exists)


os.makedirs("outputs", exist_ok=True)

# 3. Load Dataset

try:
    data = pd.read_csv("data/Mall_Customers.csv")
    print("Dataset Loaded Successfully\n")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("First 5 Records:\n")
print(data.head())

# 4. Data Preprocessing


features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nData Scaling Completed\n")

# 5. Finding Optimal K (Elbow Method)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("outputs/elbow_plot.png")
plt.close()

print("Elbow plot saved in outputs folder\n")

# 6. Apply K-Means (K = 5)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

data['Cluster'] = clusters

print("K-Means Applied Successfully\n")

# 7. Evaluate Model

score = silhouette_score(scaled_features, clusters)
print("Silhouette Score:", round(score, 3), "\n")

# 8. Visualize Clusters

plt.figure()
plt.scatter(data['Annual Income (k$)'],
            data['Spending Score (1-100)'],
            c=data['Cluster'])

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.savefig("outputs/cluster_plot.png")
plt.close()

print("Cluster plot saved in outputs folder\n")

# 9. Show Cluster Centers

centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("Cluster Centers (Age, Income, Spending Score):\n")
print(pd.DataFrame(centers,
columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']))

# 10. Save Final Dataset

data.to_csv("outputs/segmented_customers.csv", index=False)

print("\nClustered dataset saved in outputs folder")

print("\nProject Completed Successfully ✅")