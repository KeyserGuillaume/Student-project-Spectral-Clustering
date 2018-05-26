import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import generation
import graph
from random import shuffle
from sklearn.mixture import GaussianMixture

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
    D_inv = np.diag(1/d)
    return np.identity(W.shape[0]) - D_inv.dot(W)

def normalized_laplacian_sym(W):
    d = np.sum(W,axis=1)
    D_inverse_root = np.diag(1/np.sqrt(d))
    return np.identity(W.shape[0]) - D_inverse_root.dot(W.dot(D_inverse_root))

def normalized_spectral_clustering(W,k):
    Lrw = normalized_laplacian_rw(W)
    eigenvalues, eigenvectors = np.linalg.eig(Lrw)
    index_sorted = np.argsort(eigenvalues)
    k_first_eigenvectors = eigenvectors[:,index_sorted[:k]]
    kmeans = KMeans(n_clusters=k).fit(k_first_eigenvectors)
    prediction = kmeans.labels_
    clusters = {}
    for i in range(k):
        clusters[i] = np.where(prediction == i)
    return eigenvalues[index_sorted[:k]], k_first_eigenvectors, clusters

def normalized_spectral_clustering_bis(W,k):
    Lsym = normalized_laplacian_sym(W)
    eigenvalues, eigenvectors = np.linalg.eig(Lsym)
    index_sorted = np.argsort(eigenvalues)
    k_first_eigenvectors = eigenvectors[:,index_sorted[:k]]
    T = k_first_eigenvectors/(np.sqrt(np.sum(k_first_eigenvectors**2, axis=1).reshape(-1,1)))
    kmeans = KMeans(n_clusters=k).fit(T)
    prediction = kmeans.labels_
    clusters = {}
    for i in range(k):
        clusters[i] = np.where(prediction == i)
    return eigenvalues[index_sorted[:k]], k_first_eigenvectors,clusters

if __name__ == "__main__":
    # data,y = generation.gen_1d_gaussian_mixture(nbex=200)
    # data, y = generation.gen_arti(nbex=200, data_type=1)
    # data,y = generation.random_walks(4, 200, step=0.1, d=np.array([0.1, 0.2]))
    # data,y = generation.concentric_circles(2, 200)
    nbex = 1000
    generate_clusters = [generation.concentric_circles, generation.make_moons,generation.generate_cross,\
    generation.read_data_bis,generation.read_data_bis,generation.read_data_bis,generation.read_data_bis]
    paramaters = [{"nbex":nbex,"k":4},{"n_samples":nbex,"noise":0.08}, \
    {"nbex":int(nbex/2),"a":1,"eps":0.5}, {"filename":"Datasets/spiral.txt"},{"filename":"Datasets/compound.txt"},{"filename":"Datasets/d31.txt"},\
    {"filename":"Datasets/cluto-t4-8k.txt","split_char":','}]
    for i in range(6,7):
        data,y = generate_clusters[i](**paramaters[i])
        n_clusters = len(set(y.ravel()))
        S = graph.gaussian_similarity_matrix(data,5)
        W = graph.fully_connected_graph(S)
        # W = graph.k_nearest_neighbors_graph(S,10, mutual=True)
        
        if n_clusters > 7:
            colors = list(mcolors.CSS4_COLORS.keys())
        else:
            colors = "byrmcgk"
        # #Unnormalized laplacian 
        # eigenvalues, eigenvectors, clusters = unnormalized_spectral_clustering(W,n_clusters) 
        # plt.figure()
        # plt.title("Clustering spectral avec le laplacien non normalisé")
        # for i in clusters:
        #     cluster = clusters[i][0]
        #     plt.scatter(data[cluster,0], data[cluster,1], c=colors[i])
        #Normalized laplacian 
        eigenvalues, eigenvectors, clusters = normalized_spectral_clustering(W,n_clusters)
        plt.figure()
        plt.title("Clustering spectral avec le laplacien normalisé")
        for i in clusters:
            cluster = clusters[i][0]
            plt.scatter(data[cluster,0], data[cluster,1], c=colors[i])
        #Mix de Gaussienne
        gm = GaussianMixture(n_components=n_clusters)
        gm.fit(data)
        y_pred = gm.predict(data)
        plt.figure()
        plt.title("Gaussian Mixture Clustering")
        for i in range(data.shape[0]):
            plt.scatter(data[i][0],data[i][1],c=colors[y_pred[i]])

        #Kmeans
        km = KMeans(n_clusters=n_clusters)
        y_pred = km.fit_predict(data)
        plt.figure()
        plt.title("KMeans")
        for i in range(data.shape[0]):
            plt.scatter(data[i][0],data[i][1],c=colors[y_pred[i]])

    plt.show()