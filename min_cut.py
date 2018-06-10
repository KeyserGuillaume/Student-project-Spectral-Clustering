"""
Spectral Clustering may be usually solved with an eigenvalues problem, but it's
basically a min_cut problem. What I can do is to take 2 points a and b and to
find partition A and B respectively containing a and b minimizing the cut. If
we do this for several points, we should get a good clustering result between
all the results.
So first, select two points (randomly for now).
Second, calculate max b-flot
Third, deduce the min-cut
This file contains the functions necessary for doing this.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing
import time

import generation
import graph
from clustering import cluster_visualisation, GraphVisualisation


def Dijkstra_one_weight(forward, backward, s, t, visu=False, v=False):
    """
    Called one_weight because edges have weight or 1 (or zero if they don't exist)
    We find a shortest path between s and t because the Edmond and Karps algorithm 
    is polynomial in this way. Its complexity is nm^2
    """
    propagation_circles = [[s]]
    i = 0
    while t not in propagation_circles[i]:
        propagation_circles.append([])
        [[propagation_circles[i+1].append(y) for y in forward[x] if True not in [y in propagation_circles[j] for j in range(i+2)]] for x in propagation_circles[i]]
        if len(propagation_circles[i+1]) == 0:
            return "no path"
        i += 1
    if v:
        print("found a {}-long s-t path".format(i))
    if visu:
        for circle in propagation_circles:
            if len(circle) != len(np.unique(circle)):
                print(len(circle), len(np.unique(circle)))
            fig = plt.figure()
            plt.scatter(data[:,0], data[:,1], edgecolors='face', c='b')
            plt.scatter(data[circle,0], data[circle,1], edgecolors='face', c='r')
            plt.scatter(data[[s, t],0], data[[s, t],1], edgecolors='face', c='k')
            plt.show(block = False)
            time.sleep(0.5)
            plt.close(fig)
    path = [t]
    while path[0] != s:
        u = [y for y in backward[path[0]] if y in propagation_circles[i-1]]
        path = [u[0]] + path
        i -= 1
    return path

def min_cut(w, s, t, return_cut=False, verbose=False):
    """
    Compute min cut between s and t in graph w (w_{ij} = weight between i and j)
    We first compute the maximum flow between s and t and then we deduce the
    min cut.
    We use the Edmonds and Karp's algorithm for its simplicity. The idea is to
    increase the flow along a well-chosen s-t augmenting path. The path is chosen
    to minimize its length.
    """
    res_graph = w
    n = len(w)
    optimal = False
    while not optimal:
        #w_ones = (res_graph > 0)
        #path = Dijkstra_one_weight(w_ones, s, t)
        forward = [np.where(res_graph[i,:] > 0)[0] for i in range(n)]
        backward = [np.where(res_graph[:,j] > 0)[0] for j in range(n)]
        path = Dijkstra_one_weight(forward, backward, s, t)
        if path == "no path":
            optimal = True
        else:
            # decrease res_graph along path and decrease in opposite direction
            if path[-1] != t:
                print("warning : found a path ending at {} instead of t ({})".format(path[-1], t))
            if path[0] != s:
                print("warning : found a path starting at {} instead of s ({})".format(path[0], s))
            i = 0
            # find the maximum mu by which flow can be augmented
            if verbose:
                print("looking for best mu")
            max_mu = float("inf")
            while path[i] != t:
                u = path[i]
                v = path[i+1]
                max_mu = min(res_graph[u, v], max_mu)
                i += 1
            if verbose:
                print("found best mu = {}".format(max_mu))
            i = 0
            while path[i]!=t:
                u = path[i]
                v = path[i+1]
                res_graph[u, v] -= max_mu
                res_graph[v, u] += max_mu
                i += 1
    clusters = {}
    clusters[0] = [s]
    clusters[1] = [t]
    stack0 = [s]
    while len(stack0) > 0:
        x = stack0.pop()
        [stack0.append(y) for y in np.where(res_graph[x,:] > 0)[0] if y not in clusters[0]]
        [clusters[0].append(y) for y in np.where(res_graph[x,:] > 0)[0] if y not in clusters[0]]
    stack1 = [t]
    while len(stack1) > 0:
        x = stack1.pop()
        [stack1.append(y) for y in np.where(res_graph[:,x] > 0)[0] if y not in clusters[1]]
        [clusters[1].append(y) for y in np.where(res_graph[:,x] > 0)[0] if y not in clusters[1]]
    if return_cut:
        cut = sum([sum([w[i,j] for i in clusters[0]]) for j in clusters[1]])
        return clusters, cut
    else:
        return clusters
        
def g(w, q0, q, n, i, num, verbose):
    """
    w : graph matrix
    q0 : a list of possible (s,t) couples
    q : where to put the results
    n : the nb of computations required
    i : the process nb
    num : the nb of computations currently done
    """
    while num.value < n:
        s, t = q0.get()
        if verbose:
            print("Process {} : computing min_cut between {} and {} : begin.".format(i, s, t))
        if (s==t):
            print("warning : s = t")
        clusters, cut = min_cut(w, s, t, return_cut=True)
        q.put([clusters, cut])
        if verbose:
            print("Process {} : computing min_cut between {} and {} : done.".format(i, s, t))
        num.value+=1
        if verbose:
            print(num.value)
    if verbose:
        print("Process {} : finished work".format(i))
    return

def min_cut_mp_clustering(w, min_cut_nb=5, subprocessNb=4, verbose=True, return_cut=True):
    """
    Randomly select a few couples (s,t) and run min cut between them. We use 
    multi-processing so that if we stumble on a long computation (which happens
    with eps-neighbour graph sometimes), we can interrupt it when other shorter
    computations have finished clustering.
    """
    q0 = multiprocessing.Queue()
    q = multiprocessing.Queue()
    [q0.put([np.random.randint(0, len(w)-1), np.random.randint(0, len(w)-1)]) for i in range(3*min_cut_nb)]
    jobs = list()
    num = multiprocessing.Value('i', 0)
    for i in range(subprocessNb):
        p = multiprocessing.Process(target = g, args=(w, q0, q, min_cut_nb+1, i, num, verbose,))
        p.start()
        jobs.append(p)
    best_cut = float('inf')
    best_clusters = {}
    for i in range(min_cut_nb):
        if verbose:
            print("getting result {}".format(i))
        clusters, cut = q.get()
        if cut < best_cut and len(clusters[0]) > 1 and len(clusters[1]) > 1 and len(clusters[0])+len(clusters[1])==len(w):
            best_cut = cut
            best_clusters = clusters
    [jobs[i].terminate() for i in range(subprocessNb)]
    if verbose:
        print("min cut computations done")
    if best_clusters=={}:
        print("min_cut_mp_clustering : Failed to partition graph")
    if return_cut:
        return best_clusters, best_cut
    return best_clusters
    
def dynamic_programming_for_multiclass(k, NN):
    """
    V_{k, N} = \max_{n_1 + n_2 + ... n_{k-1} = N}  \sum_{i=1}^{k-1} log(1-(p_i^2 + q_i^2)^{n_i}
             = \max_{n_1} log(1-(p_i^2 + q_i^2)^{n_1}) + V(k-1, N-n_1)
    """
    p = [1/(i+1) for i in range(k)]
    V = -np.inf*np.ones((k-1, NN+1))
    V = np.vstack((np.zeros((1, NN+1)), V))
    Policy = np.zeros((k, NN+1))
    for i in range(1, k):
        for N in range(1, NN+1):
            for n in range(1, N+1):
                V_n = np.log(1 - (p[i]**2 + (1-p[i])**2)**n) + V[i-1,N-n]
                if V_n >= V[i, N]:
                    V[i, N] = V_n
                    Policy[i, N] = n
    solution = [int(Policy[k-1, NN])]
    N = NN
    for i in range(k-2, 0, -1):
        N -= solution[0]
        solution = [int(Policy[i, N])] + solution
    print(solution)
    return Policy

def _min_cut_multiclass_clustering(w, k, N, min_cut_nb_list):
    """
    does clustering in two classes recursively
    Returns a list of possible k-sized clusterings for graph matrix w
    First we cluster in two what we have (all of w)
    Then, we have possible cluster repartitions given by:
    (1, k-1), (2, k-2), (3, k-3), ..., (k-1, 1)
    For each of them the function calls itself to give yet again possible
    clusters and the cut values
    Those are then joined for each possibility of each cluster
    And then the whole lot is returned as result of the function 
    """
    print("Hey !")
    if k==1:
        return [[{0: np.arange(len(w))}, 0]]
    clusters_mixed_classes, cut = min_cut_mp_clustering(w, verbose = False, min_cut_nb = 20, return_cut=True)
    cluster1 = clusters_mixed_classes[0]
    cluster2 = clusters_mixed_classes[1]
    w_reduced1 = w[cluster1,:][:,cluster1]
    w_reduced2 = w[cluster2,:][:,cluster2]
    list_possible_clusters = []
    for i in range(1, k):
        possible_clusters_in_1 = _min_cut_multiclass_clustering(w_reduced1, i, N, min_cut_nb_list)
        possible_clusters_in_2 = _min_cut_multiclass_clustering(w_reduced2, k-i, N, min_cut_nb_list)
        for clusters1, cut1 in possible_clusters_in_1:
            for clusters2, cut2 in possible_clusters_in_2:
                joined_clusters = {}
                for j in range(i):
                    joined_clusters[j] = np.array(cluster1)[clusters1[j]]
                for j in range(i, k):
                    joined_clusters[j] = np.array(cluster2)[clusters2[j-i]]
        list_possible_clusters.append([joined_clusters, cut + cut1 + cut2])
    return list_possible_clusters

def min_cut_multiclass_clustering(w, k, N = 30):
    #min_cut_nb_list = dynamic_programming_for_multiclass(k, N)
    min_cut_nb_list = None
    possible_clusters = _min_cut_multiclass_clustering(w, k, 30, min_cut_nb_list)
    best_cut = float('inf')
    best_clusters = {}
    for clusters, cut in possible_clusters:
        if cut < best_cut:
            best_cut = cut
            best_clusters = clusters
    return clusters

def test_clustering(data, y, full = False, s = None, t = None):
    n_clusters = len(np.unique(y))
    epsilon = graph.suggest_epsilon(data)
    if full:
        graph_methods = [graph.eps_neighborhood_graph(eps = epsilon, allow_override = True),
                         graph.k_nearest_neighbors_graph(k = 15, mutual = True, allow_override = True)]
                         #graph.fully_connected_graph()]
    else:
        graph_methods = [graph.k_nearest_neighbors_graph(k = 15, mutual = True, allow_override = True)]
    print("computing similarity matrix")
    S = graph.gaussian_similarity_matrix(data)
    if s is None:
        s = np.random.randint(0, len(S)-1)
    if t is None:
        t = np.random.randint(0, len(S)-1)
    if s==t:
        print("warning : s et t identiques")
    for graph_meth in graph_methods:
        print("computing " + graph_meth.name)
        w = graph_meth(S)
        clusters = min_cut(w, s, t)
        cluster_visualisation(data, clusters, "min-cut max-flow with " + graph_meth.name)

if __name__=="__main__":
    #data, y = generation.gen_arti(nbex = 100, data_type = 0, epsilon = 0.1)
    #data, y = generation.gen_arti(nbex = 500, data_type = 1, epsilon = 0.1)
    #data, y = generation.concentric_circles(k = 2, nbex = 500, eps = 0.05)
    data, y = generation.read_data_bis("Datasets/spiral.txt", split_char="\t")
    
    plt.title("raw data")
    plt.scatter(data[:,0], data[:,1], edgecolors='face')
    plt.show()
    #test_clustering(data, y, full=True)
    
    S = graph.gaussian_similarity_matrix(data)
    graph_meth = graph.k_nearest_neighbors_graph(k = 15, mutual = True, allow_override = True)
    #graph_meth = graph.eps_neighborhood_graph(eps = graph.suggest_epsilon(data), allow_override=True)
    w = graph_meth(S)
    #clusters = min_cut_mp_clustering(w)
    clusters = min_cut_multiclass_clustering(w, 3, N = 80)
    #s = np.random.randint(0, len(w)-1)
    #t = np.random.randint(0, len(w)-1)
    #clusters = min_cut(w, s, t)
    cluster_visualisation(data, clusters, "clustering with max-flow min-cut")
    #GraphVisualisation(data, y, graph_meth)