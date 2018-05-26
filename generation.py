import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs

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

 # recyclee du tp 3 de malap
def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//2))
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex//2))
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),int(nbex//4))))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(int(nbex//2))))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on melange les donnees
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y
    
def random_walks(k, nbex, step=0.005, noise=0.005, d=None):
    """
    L'idee est d'avoir k marches aleatoires
    """
    data = np.array([]).reshape(-1,2)
    y = np.array([])
    timesToWalk = nbex/k*np.ones(nbex)
    for i in range(k):
        x = np.random.rand(2,1).reshape(-1,2)
        if d.any():
            d = d.reshape(-1,2)
        else:
            d = (np.random.rand(2,1)-0.5).reshape(-1,2)
        data = np.vstack((data,x))
        for t in range(int(timesToWalk[i]-1)):
            x+=step*d+noise*(np.random.rand(2,1)-0.5).reshape(-1,2)
            data = np.vstack((data,x))
        y = np.concatenate((y,i*np.ones(timesToWalk[i])))
    idx = np.random.permutation(nbex)
    data=data[idx]
    y=y[idx]
    return data, y.reshape(-1,1)
    
def concentric_circles(k, nbex):
    data = np.array([]).reshape(-1,2)
    y = np.array([])
    radius_list = 1/3+np.arange(0,4*k)/(12*k)
    for i in range(k):
        for j in range(int(nbex/k)):
            r1 = radius_list[3*i]
            r2 = radius_list[3*i+1]
            theta = 6.28*float(np.random.rand(1,1))
            r = r2 + float(np.random.rand(1,1))*(r1-r2)
            a = r*np.cos(theta)
            b = r*np.sin(theta)
            data = np.vstack((data, np.array([[a,b]])))
        y = np.concatenate((y,i*np.ones(int(nbex/k))))
    idx = np.random.permutation(nbex)
    data=data[idx]
    y=y[idx]
    return data, y.reshape(-1,1)

def generate_cross(nbex,a,eps):
    x1 = np.random.uniform(-5,5,(nbex,1))
    noise = np.random.normal(0,eps,(nbex,1))
    y1 = a*x1+noise
    data = np.hstack((x1,y1))
    x2 = np.random.uniform(-5,5,(nbex,1))
    noise = np.random.normal(0,eps,(nbex,1))
    y2 = -a*x2 + noise
    data = np.vstack((data,np.hstack((x2,y2))))
    labels = np.vstack((np.ones((nbex,1)),-np.ones((nbex,1))))
    p = np.random.permutation(2*nbex)
    data = data[p]
    labels = labels[p]
    return data,labels

def read_data(namedata,namelabels):
    with open(namedata) as f:
        lines = list(map(str.rstrip, f.readlines()))
        datas = list(map(lambda x:x.split(' '), lines))
        datas_treated = []
        for data in datas:
            datas_treated.append([int(x) for x in data if x != ''])
    with open(namelabels) as f:
        lines = list(map(str.rstrip, f.readlines()))
        datas = list(map(lambda x:x.split(' '), lines))
        labels = []
        for data in datas:
            labels.append([int(data[0])])
    return np.asarray(datas_treated),np.asarray(labels)

def read_data_bis(filename,split_char=' '):
    with open(filename) as f:
        lines = list(map(str.rstrip, f.readlines()))
        datas = list(map(lambda x:x.split(split_char), lines))
        datas_treated = []
        labels = []
        for data in datas:
            datas_treated.append([float(data[0]),float(data[1])])
            labels.append(int(data[2]))
    return np.asarray(datas_treated),np.asarray(labels)