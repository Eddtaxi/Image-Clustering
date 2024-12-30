import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
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

# Estimazione del bandwidth per MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Applicazione dell'algoritmo MeanShift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels_MS = ms.labels_
cluster_centers_MS = ms.cluster_centers_

# Determinazione del numero di cluster
labels_unique_MS = np.unique(labels_MS)
n_clusters_MS = len(labels_unique_MS)

print("Numero di cluster stimati: %d" % n_clusters_MS)

# Visualizzazione dei risultati del clustering
plt.subplot(1, 2, 2)
# Usa il colore per ciascun cluster
plt.scatter(X[:, 0], X[:, 1], c=labels_MS, cmap='viridis', s=15)
# Visualizza i centri dei cluster
plt.scatter(cluster_centers_MS[:, 0], cluster_centers_MS[:, 1], c='red', marker='x', s=100, label='Centri dei Cluster')
plt.title('Clustering con MeanShift')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')
plt.legend()

# Mostra i grafici
plt.tight_layout()
plt.show()
