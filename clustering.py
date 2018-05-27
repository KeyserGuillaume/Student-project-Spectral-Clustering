import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.neighbors import kneighbors_graph
import hdbscan
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import generation
import graph
from random import shuffle
import sklearn
if sklearn.__version__=="0.17":
    from sklearn.mixture import GMM as GaussianMixture
else:
    from sklearn.mixture import GaussianMixture

def unnormalized_laplacian(W):
    d = np.sum(W,axis=1)
    D = np.diag(d)
    return D-W

class unnormalized_spectral_clustering:
    def __init__(self):
        self.name = "Unnormalised spectral clustering"
    def __call__(self, W, k):
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
    #print(d)
    D_inv = np.diag(1/d)
    return np.identity(W.shape[0]) - D_inv.dot(W)

def normalized_laplacian_sym(W):
    d = np.sum(W,axis=1)
    D_inverse_root = np.diag(1/np.sqrt(d))
    return np.identity(W.shape[0]) - D_inverse_root.dot(W.dot(D_inverse_root))

class normalized_spectral_clustering:
    def __init__(self):
        self.name = "Normalised spectral clustering"
    def __call__(self, W, k):
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

class normalized_spectral_clustering_bis:
    def __init__(self):
        self.name = "Normalised spectral clustering bis"
    def __call__(self, W, k):
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

class gaussian_mixture:
    def __init__(self):
        self.name = "Gaussian Mixture"
    def __call__(self, data, n):
        gm = GaussianMixture(n_components = n)
        gm.fit(data)
        y_pred = gm.predict(data)
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters

class mean_shift:
    def __init__(self):
        self.name = "Mean Shift"
    def __call__(self, data, n):
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        y_pred = ms.predict(data)
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters

class agglomerative_clustering:
    def __init__(self):
        self.name = "Agglomerative Clustering"
    def __call__(self, data, n):
        knn_graph = kneighbors_graph(data, 30, include_self=False)
        model = AgglomerativeClustering(linkage='average',
                                        connectivity=knn_graph,
                                        n_clusters=n)
        model.fit(data)
        y_pred = model.labels_
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters

class dbscan:
    def __init__(self):
        self.name = "DBSCAN"
    def __call__(self, data, n):
        db = DBSCAN(eps=3.6, min_samples=8).fit(data)
        y_pred =  db.labels_
        print(len(np.unique(y_pred)))
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters    

class Hdbscan:
    def __init__(self):
        self.name = "HDBSCAN"
    def __call__(self, data, n):
        clusterer = hdbscan.HDBSCAN(min_cluster_size = 5)
        y_pred = clusterer.fit_predict(data)
        print(len(np.unique(y_pred)))
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters    

class k_means:
    def __init__(self):
        self.name = "k-means"
    def __call__(self, data, n):
        km = KMeans(n_clusters=n)
        y_pred = km.fit_predict(data)
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        return clusters

class affinity_propagation:
    def __init__(self):
        self.name = "Affinity Propagation"
    def __call__(self, data, n):
        af = AffinityPropagation().fit(data)
        cluster_centers_indices = af.cluster_centers_indices_
        y_pred = af.labels_
        clusters = {i:np.where(y_pred==i) for i in np.unique(y_pred)}
        print(len(np.unique(y_pred)))
        return clusters
        
def cluster_visualisation(data, clusters, title):
    """
    visualisation dans R^2     
    """
    if len(clusters) > 7:
        colors = list(mcolors.CSS4_COLORS.keys())
    else:
        colors = "byrmcgk"    
    plt.figure()
    plt.title(title)
    for i in clusters:
        cluster = clusters[i][0]
        plt.scatter(data[cluster,0], data[cluster,1], c=colors[i], edgecolors='face')
    plt.show()

def GraphVisualisation(data, y, graph_meth = None):
    """
    Visualise les clusters ordonnes dans la matrice de similarite
    """
    if graph_meth==None:
        graph_meth = graph.fully_connected_graph()
    data = np.vstack([data[np.where(y==i)] for i in np.unique(y)])
    S = graph.gaussian_similarity_matrix(data)
    W = graph_meth(S)
    plt.title(graph_meth.name)
    plt.imshow(W, interpolation="nearest", cmap="gray")
    L = normalized_laplacian_rw(W)
    plt.figure()
    plt.title("Normalized laplacian")
    plt.imshow(L, interpolation="nearest", cmap="gray")
    plt.show()

def test_clustering(data, y, full = False, comparison = False):
    """
    Avec full on essaie les 9 possibilites (3 graphes et 3 laplaciens)
    Avec comparison on compare aux methodes alternatives de clustering
    """
    n_clusters = len(np.unique(y))
    epsilon = graph.suggest_epsilon(data)
    if full:
        graph_methods = [graph.eps_neighborhood_graph(eps = epsilon, allow_override = True),
                         graph.k_nearest_neighbors_graph(k = 15, mutual = True, allow_override = True),
                         graph.fully_connected_graph()]
        clustering_methods = [unnormalized_spectral_clustering(),
                              normalized_spectral_clustering(),
                              normalized_spectral_clustering_bis()]
    else:
        graph_methods = [graph.k_nearest_neighbors_graph(k = 15, mutual = True, allow_override = True)]
        clustering_methods = [normalized_spectral_clustering()]
    if comparison:
        alternative_methods = [gaussian_mixture(),
                               k_means(),
                               mean_shift(),
                               agglomerative_clustering(),
                               dbscan(),
                               Hdbscan()]
                               #affinity_propagation()]
    else:
        alternative_methods = []
    
    for meth in alternative_methods:
        clusters = meth(data, n_clusters)
        cluster_visualisation(data, clusters, meth.name)
        
    print("computing similarity matrix")
    S = graph.gaussian_similarity_matrix(data)
    for graph_meth in graph_methods:
        print("computing " + graph_meth.name)
        W = graph_meth(S)
        for clustering_meth in clustering_methods:
            print("computing clusters with " + clustering_meth.name)
            _, _, clusters = clustering_meth(W, n_clusters)
            cluster_visualisation(data, clusters, clustering_meth.name + " with " + graph_meth.name)

def test_USPS_data():
    data, y = generation.load_usps("USPS_train.txt")
    data = data[0:1000]
    y = y[0:1000]
    S = graph.gaussian_similarity_matrix(data)
    W = graph.k_nearest_neighbors_graph(k = 10, mutual = False)(S)
    _, _ , clusters = normalized_spectral_clustering()(W, len(np.unique(y)))
    fig = plt.figure()
    fig.suptitle("Number repartition in clusters", fontsize=16)
    for i in range(10):
        ax = plt.subplot(4,3,i+1)
        ax.set_title("group "+str(i))
        plt.hist(y[clusters[i][0]], bins=range(11))
        plt.xticks(range(11))
    plt.show()

if __name__ == "__main__":
#    data, y = generation.gen_arti(nbex = 500, data_type = 1, epsilon = 0.3)
#    data, y = generation.random_walks(nbex = 500, parallel=True)
#    data, y = generation.random_walks(nbex = 500)
#    data, y = generation.concentric_circles(nbex = 500)
#    data, y = generation.generate_cross(nbex = 500)
#    data, y = generation.read_data_bis("Datasets/spiral.txt", split_char="\t")
    data, y = generation.read_data_bis("Datasets/cluto-t4-8k.txt", split_char=",")
    test_clustering(data, y, full = False, comparison = True)
#    test_USPS_data()
#    GraphVisualisation(data, y, graph.eps_neighborhood_graph(eps = graph.suggest_epsilon(data), allow_override=True))
#    GraphVisualisation(data, y)
#    GraphVisualisation(data, y, graph.k_nearest_neighbors_graph(k = 10, mutual = False))
    