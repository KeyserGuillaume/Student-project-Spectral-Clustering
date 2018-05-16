import numpy as np

def gen_1d_gaussian_mixture(centers = [-3,-1,1,3],sigma=0.1,nbex=1000):
    data = np.array([])
    for i in centers:
        data = np.concatenate((data,np.random.normal(i,sigma,int(nbex/4.))))
    y = np.array([])
    for i in range(4):
        y = np.concatenate((y,i*np.ones(int(nbex/4.))))
    idx = np.random.permutation(nbex)
    data=data[idx]
    y=y[idx]
    return data.reshape(-1,1),y.reshape(-1,1)
