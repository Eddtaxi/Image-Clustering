import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Creazione di un dataset di esempio
X, _ = make_blobs(n_samples=900, centers=4, cluster_std=0.50, random_state=0)

# Visualizzazione dei dati originali
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', s=15)
plt.title('Dati Originali')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')

# Applicazione dell'algoritmo K-Means
k = 4  # Numero di cluster desiderati
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)

# Esegui il clustering
kmeans.fit(X)
labels_KM = kmeans.labels_
cluster_centers_KM = kmeans.cluster_centers_

# Visualizzazione dei risultati del clustering
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_KM, cmap='viridis', s=15)
plt.scatter(cluster_centers_KM[:, 0], cluster_centers_KM[:, 1], c='red', marker='x', s=100, label='Centri Cluster')
plt.title(f'K-Means Clustering con {k} Cluster')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')
plt.legend()

# Mostra i grafici
plt.tight_layout()
plt.show()
