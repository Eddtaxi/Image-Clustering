
import matplotlib.pyplot as plt  
import numpy as np
from skimage.io import imread  
from sklearn.cluster import MeanShift, estimate_bandwidth


# Caricamento dell'immagine da file
immagine = imread('Lenna.png')  
w,h,c = immagine.shape
array_immagine = immagine.reshape(-1, 3)  

# L'immagine in formato originale è una matrice tridimensionale, 
# l'algoritmi di clustering, richiede che i dati siano organizzati in una forma 2D, 
# dove ogni riga rappresenta un singolo campione (in questo caso un pixel dell'immagine)
# e le colonne rappresentano le caratteristiche (i valori RGB).

array_immagine = np.float32(array_immagine)

# Le immagini in formato RGB sono generalmente rappresentate con valori interi tra 0 e 255 per ciascun colore. 
# Quando si effettuano calcoli complessi come il clustering, questi valori interi potrebbero non essere abbastanza precisi. 
# Usare np.float32 permette di avere valori decimali,che sono più adatti per calcoli accurati e veloci,come nel caso di algoritmi come MeanShift.

# Calcolo della bandwidth per il clustering MeanShift
bandwidth = estimate_bandwidth(array_immagine, quantile=0.3, n_samples=500) #quantile=0.05,quantile=0.2,quantile=0.3

# Creazione dell'oggetto MeanShift e applicazione del clustering
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True) 
meanshift.fit(array_immagine)  


labels = meanshift.labels_  
centri_cluster = meanshift.cluster_centers_.astype(np.uint8)  

labels_unique = np.unique(labels) 
n_clusters_ = len(labels_unique)  

# labels è un array che contiene l'etichetta del cluster a cui ogni pixel dell'immagine è stato assegnato dopo il clustering. 
# Ad esempio, se hai 10000 pixel nell'immagine e 5 cluster,ogni pixel avrà un valore corrispondente al cluster a cui appartiene.
# np.unique(labels) restituisce un array contenente i valori unici di etichetta 
# len(labels_unique) conta i cluster

print(f"Number of estimated clusters : {n_clusters_}")  # Stampa il numero di cluster stimati

# Ricostruzione dell'immagine clusterizzata
# Mappa le etichette ai centri dei cluster per creare una nuova immagine
res = centri_cluster[labels].reshape(w, h, c)  # Riassembla l'immagine dai centri dei cluster

# Visualizzazione dell'immagine originale e dell'immagine clusterizzata
plt.subplot(121)  
plt.imshow(immagine) 
plt.title('immagine originale')  

plt.subplot(122)  
plt.imshow(res)  
plt.title(f'Mean-shift cluster con {n_clusters_} clusters')  # Aggiungi il titolo con il numero di cluster
plt.show()  
