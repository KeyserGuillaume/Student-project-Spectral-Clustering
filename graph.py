import numpy as np

def gaussian_similarity_matrix(x,sigma=1):
    n = x.shape[0]
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            S[i][j] = np.exp(-(np.linalg.norm(x[i]-x[j])**2)/(2*sigma**2)) 
            S[j][i] = S[i][j]
    return S

def eps_neighborhood_graph(S,eps):
    n = S.shape[0]
    mat_adj = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            mat_adj[i][j] = 1 if S[i][i] > eps else 0
            mat_adj[j][i] = mat_adj[i][j]
    return mat_adj
    
def k_nearest_neighbors(k,similarity):
    n = similarity.shape[0]
    sorted_similarity = similarity[similarity[:,1].argsort()]
    #Returns the index of the k nearest neighbors
    return sorted_similarity[n-k+1:,0]

def k_nearest_neighbors_graph(S,k,mutual=True):
    n = S.shape[0]
    mat_adj = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            neighbors_i = np.asarray([[idx,s] for (idx,s) in enumerate(S[i,:]) if idx!=i]).reshape(-1,2)
            neighbors_j = np.asarray([[idx,s] for (idx,s) in enumerate(S[j,:]) if idx!=j]).reshape(-1,2)
            knn_i = k_nearest_neighbors(k,neighbors_i)
            knn_j = k_nearest_neighbors(k,neighbors_j)
            if mutual:
                mat_adj[i][j] = S[i][j] if (j in knn_i and i in knn_j) else 0
                mat_adj[j][i] = mat_adj[i][j]
            else:
                mat_adj[i][j] = S[i][j] if (j in knn_i or i in knn_j) else 0
                mat_adj[j][i] = mat_adj[i][j]
    return mat_adj

def fully_connected_graph(S):
    n = S.shape[0]
    mat_adj = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            mat_adj[i,j] = S[i,j]
            mat_adj[j,i] = S[i,j]
    return mat_adj