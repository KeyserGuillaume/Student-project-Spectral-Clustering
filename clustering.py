import numpy as np
from sklearn.cluster import KMeans
import generation
import graph

def unnormalized_laplacian(W):
    d = np.sum(W,axis=1)
    D = np.diag(d)
    return D-W

def unnormalized_spectral_clustering(W,k):
    L = unnormalized_laplacian(W)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    index_sorted = np.argsort(eigenvalues)
    k_first_eigenvectors = eigenvectors[:,index_sorted[:k]]
    kmeans = KMeans(n_clusters=k).fit(k_first_eigenvectors)
    prediction = kmeans.labels_
    clusters = {}
    for i in range(k):
        clusters[i] = np.where(prediction == i)
    return eigenvalues[index_sorted[:k]], k_first_eigenvectors,clusters

def normalized_laplacian_rw(W):
    d = np.sum(W,axis=1)
    print(d)
    D_inv = np.diag(1/d)
    return np.identity(W.shape[0]) - D_inv.dot(W)

def normalized_laplacian_sym(W):
    d = np.sum(W,axis=1)
    D_inverse_root = np.diag(1/np.square(d))
    return np.identity(W.shape[0]) - D_inverse_root.dot(W.dot(D_inverse_root))

def normalized_spectral_clustering(W,k):
    Lrw = normalized_laplacian_rw(W)
    eigenvalues, eigenvectors = np.linalg.eig(Lrw)
    index_sorted = np.argsort(eigenvalues)
    k_first_eigenvectors = eigenvectors[:,index_sorted[:k]]
    kmeans = KMeans(n_clusters=k).fit(k_first_eigenvectors)
    prediction = kmeans.labels_
    print(prediction)
    clusters = {}
    for i in range(k):
        clusters[i] = np.where(prediction == i)
    return eigenvalues[index_sorted[:k]], k_first_eigenvectors, clusters

def normalized_spectral_clustering_bis(W,k):
    Lsym = normalized_laplacian_sym(W)
    eigenvalues, eigenvectors = np.linalg.eig(Lsym)
    index_sorted = np.argsort(eigenvalues)
    k_first_eigenvectors = eigenvectors[:,index_sorted[:k]]
    kmeans = KMeans(n_clusters=k).fit(k_first_eigenvectors)
    prediction = kmeans.labels_
    clusters = {}
    for i in range(k):
        clusters[i] = np.where(prediction == i)
    return eigenvalues[index_sorted[:k]], k_first_eigenvectors,clusters

if __name__ == "__main__":
    # data,y = generation.gen_1d_gaussian_mixture(nbex=200)
    data, y = generation.gen_arti(nbex=200, data_type=1)
    # data,y = generation.random_walks(4, 200, step=0.1, d=np.array([0.1, 0.2]))
    S = graph.gaussian_similarity_matrix(data)
    W_fullyconnected = graph.fully_connected_graph(S)
    W_neighbors = graph.k_nearest_neighbors_graph(S,10, mutual=False)
    
    #Unnormalized laplacian 
    # eigenvalues, eigenvectors, clusters = unnormalized_spectral_clustering(W_neighbors,4) incompatible with display below
    # eigenvalues, eigenvectors, clusters = normalized_spectral_clustering(W_neighbors,4)
    eigenvalues, eigenvectors, clusters = normalized_spectral_clustering_bis(W_neighbors,4)
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.scatter(data[:,0], data[:,1], edgecolors='face')
    plt.figure(2)
    colors = "byrmcgkw"
    for i in clusters:
        cluster = clusters[i][0]
        plt.scatter(data[cluster,0], data[cluster,1], c=colors[i], edgecolors='face')
    plt.show()