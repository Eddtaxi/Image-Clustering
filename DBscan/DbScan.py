import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.io import imread
from skimage.transform import resize

# Carica l'immagine
image_path = "jesus.jpg"  
img = imread(image_path)
img_resized = resize(img, (50, 50))  # Ridimensiona l'immagine per velocizzare i calcoli
w, h, c = img_resized.shape

# Riorganizza i dati dell'immagine in un array 2D
image_array = img_resized.reshape((-1, c))

# Converte in float32 per una migliore precisione nei calcoli
image_array = np.float32(image_array)

# Applica l'algoritmo DBSCAN
eps = 0.1 # Distanza massima per considerare due punti vicini
min_samples = 100   # Numero minimo di punti per definire un cluster
db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

# Esegui il clustering
labels = db.fit_predict(image_array)

# Converte le etichette in una mappa di colori
unique_labels = np.unique(labels)
n_clusters = len(unique_labels)  # Esclude il rumore
segmented_image = np.zeros_like(image_array)

# Assegna un colore casuale a ciascun cluster
colors = np.random.rand(len(unique_labels), c)
for label in unique_labels:
    if label != -1:  # Ignora il rumore
        segmented_image[labels == label] = colors[label]

# Ridimensiona i dati per visualizzare l'immagine segmentata
segmented_image = segmented_image.reshape((w, h, c))

# Mostra l'immagine originale e quella segmentata
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_resized)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f"Segmented Image (DBSCAN)\nClusters: {n_clusters}")
plt.show()
