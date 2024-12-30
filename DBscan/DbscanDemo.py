
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Creazione di un dataset di esempio
X, _ = make_blobs(n_samples=900, centers=9, cluster_std=0.50, random_state=0)

# Visualizzazione dei dati generati
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', s=15)
plt.title('Dati Originali')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')

# Applicazione dell'algoritmo DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
labels = db.fit_predict(X)

# Visualizzazione dei risultati del clustering
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=15)
plt.title('Clustering con DBSCAN')
plt.xlabel('Asse X')
plt.ylabel('Asse Y')

# Mostra i grafici
plt.show()
