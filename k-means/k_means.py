import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans

# Carica l'immagine RGB
img = imread('mountain.jpg') 

# Trasforma l'immagine in una matrice di pixel RGB
h, w, c = img.shape
image_array = img.reshape((-1, 3))

# Crea un oggetto KMeans
n_clusters = 3  # Numero desiderato di cluster
kmeans = KMeans(n_clusters=n_clusters, n_init='auto')

# Esegui il clustering sui pixel dell'immagine
kmeans.fit(image_array)
labels = kmeans.labels_  # Etichette per ogni pixel
cluster_centers = kmeans.cluster_centers_.astype(np.uint8)  # Centroidi dei cluster

# Crea l'immagine clusterizzata utilizzando i colori dei centroidi
clustered_image = cluster_centers[labels].reshape(h, w, c)

# Mostra l'immagine originale e quella clusterizzata
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Immagine Originale')
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title(f'Immagine Clusterizzata ({n_clusters} cluster)')
plt.imshow(clustered_image)
plt.show()