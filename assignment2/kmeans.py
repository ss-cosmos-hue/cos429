import numpy as np

def calculate_labels(x, centroids):
    """ calculate labels. for every point in x, assign label which is closest to centroids. 
    
    
    Arguments
        x: numpy of size (N, D)
        centroids: numpy array of size (K, D)
        
    Output
        labels : numpy of size (N,)
    """
    x, centroids = x.copy(), centroids.copy()
    N, D = x.shape
    K, D = centroids.shape
    
    ### YOUR CODE STARTS HERE ###
    
    # def calc_diff(vec1,vec2):
    #     return np.linalg.norm(vec1,vec2)
    # table = calc_diff(x,centroids)
    labels = np.array(N)
    table = np.zeros((N,K))    
    
    # def diff(centvec):
    #     return np.linalg.norm((x-centvec),axis = 1)
    
    # table = np.transpose(np.apply_along_axis(diff,axis = 1,arr=centroids))
    
    for i in range(K):
        print("process",i)
        table[:,i] = np.linalg.norm((x-centroids[i]),axis=1)#this process seems to take time
        
    # table = np.linalg.norm(([x]*K-centroids),axis=1)
    
    labels = np.argmin(table,axis=1)     
    ### YOUR CODE ENDS HERE ###
    
    return labels

def calculate_centroid(x, labels, K):
    """ Calculate new centroid using labels and x. 
    
    Arguments
        x: numpy of size (N, D)
        labels: numpy of size (N,)
        K: integer.
    Output
        centroids: numpy array of size (K, D)
    """
    x, labels = x.copy(), labels.copy()
    N, D = x.shape
    N = labels.shape
    centroids = np.zeros((K, D))
    
    ### YOUR CODE STARTS HERE ###
    # rangeK = np.arange(K)
    # def update(i):
    #     return np.mean(x[labels == i],axis = 0)
    # centroids = np.apply_along_axis(rangeK)
    for i in range(K):
        # print("centnum",i)
        centroids[i] = np.mean(x[labels == i],axis = 0)
    ### YOUR CODE ENDS HERE ###
    
    return centroids


def kmeans(x, K, niter, seed=123):
    """
    x: array of shape (N, D)
    K: integer
    niter: integer

    labels: array of shape (height*width, )
    centroids: array of shape (K, D)
    
    Note: Be careful with the size of numpy array!
    """
    print("aa")
    np.random.seed(seed)
    N,D = np.shape(x)
    unique_colors = np.unique(x.reshape(-1, D), axis=0)
    print("a")#takes time like 1minute
    
    idx = np.random.choice(len(unique_colors), K, replace=False)

    # Randomly choose centroids
    centroids = unique_colors[idx, :]
    print("b")
    

    # Initialize labels
    labels = np.zeros((x.shape[0], ), dtype=np.uint8)
    print("c")

    ### YOUR CODE STARTS HERE ###
    for _ in range(niter):
        print(_)
        labels = calculate_labels(x, centroids)        
        centroids = calculate_centroid(x, labels, K)
    
    ### YOUR CODE ENDS HERE ###
    
    return labels, centroids
