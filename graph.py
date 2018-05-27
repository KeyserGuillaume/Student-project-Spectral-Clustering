import numpy as np

def gaussian_similarity_matrix(x, sigma=1):
    n = x.shape[0]
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            S[i][j] = np.exp(-(np.linalg.norm(x[i]-x[j])**2)/(2*sigma**2)) 
            S[j][i] = S[i][j]
    return S

def suggest_epsilon(x, sigma=5):
    n = x.shape[0]
    min_link = 1000000
    for i in range(n):
        for j in range(i+1):
            min_link = min(min_link, np.exp(-(np.linalg.norm(x[i]-x[j])**2)/(2*sigma**2)))
    return 1.7*min_link

class eps_neighborhood_graph:
    def __init__(self, eps = None, allow_override = False):
        self.eps = eps
        self.name = "epsilon-neighborhood graph"
        self.allow_override = allow_override
    def __call__(self, S):
        n = S.shape[0]
        mat_adj = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                mat_adj[i][j] = 1 if S[i][j] > self.eps else 0
                mat_adj[j][i] = mat_adj[i][j]
        if self.allow_override:
            if 0 in [np.sum(mat_adj[:,i]) for i in range(n)]:
                self.eps *= 0.8
                print("warning : overriding value of epsilon to avoid divisions by zero when computing graph laplacian")
                return self.__call__(S)
        return mat_adj
    
def k_nearest_neighbors(k,similarity):
    n = similarity.shape[0]
    sorted_similarity = similarity[similarity[:,1].argsort()]
    #Returns the index of the k nearest neighbors
    return sorted_similarity[n-k+1:,0]

class k_nearest_neighbors_graph:
    def __init__(self, k = 10, mutual = True, allow_override = False):
        self.k = k
        self.mutual = mutual
        self.name = "k-nearest neighbors graph"
        self.allow_override = allow_override
    def __call__(self, S):
        n = S.shape[0]
        mat_adj = np.zeros((n,n))
        knn_matrix = []
        for i in range(n):
            neighbors_i = np.asarray([[idx,s] for (idx,s) in enumerate(S[i,:]) if idx!=i]).reshape(-1,2)
            knn_matrix.append(k_nearest_neighbors(self.k,neighbors_i).ravel())
        for i in range(n):
            for j in range(i):
                if self.mutual:
                    mat_adj[i][j] = S[i][j] if (j in knn_matrix[i] and i in knn_matrix[j]) else 0
                    mat_adj[j][i] = mat_adj[i][j]
                else:
                    mat_adj[i][j] = S[i][j] if (j in knn_matrix[i] or i in knn_matrix[j]) else 0
                    mat_adj[j][i] = mat_adj[i][j]
        if self.allow_override:
            if 0 in [np.sum(mat_adj[:,i]) for i in range(n)]:
                self.k += 5
                print("warning : overriding value of k to avoid divisions by zero when computing graph laplacian")
                return self.__call__(S)
        return mat_adj

class fully_connected_graph:
    def __init__(self):
        self.name = "fully connected graph"
    def __call__(self, S):
        n = S.shape[0]
        mat_adj = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1):
                mat_adj[i,j] = S[i,j]
                mat_adj[j,i] = S[i,j]
        return mat_adj
