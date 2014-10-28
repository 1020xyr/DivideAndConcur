# Module to embed data using divide and concur

"""
For more info:
Dissertation: Gravel, Simon: Using symmmetries to solve asymmetric problems,
    Cornell University, August 2008
Gravel, S., and Elser, V.: Divide and concur:
    A general approach to constraint satisfaction, Physical Review E 78(3),
    APS, 36706, 2008
Elser, V., Rankenburg, I., and Thibault, P.: Searching with iterated maps,
    Proceedings of the National Academy of Sciences 104(2),
    National Acad Sciences, 418, 2007
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sys import stdout

def knn(data, K):
    """
    k-nearest neighbors wrapper.

    Parameters
    ----------
    data: array-like
        Input data.
    K: int
        k from knn.
    """

    neighbors = KNN(n_neighbors=K)
    neighbors.fit(data, data)
    return neighbors

def initialize_embeddings(data, n_idx, D=2):
    """
    Function to initialize embeddings and broadcast them into a divided one.

    Parameters
    ----------
    data: array-like
        Input data.
    n_idx: list of list of ints
        Mapping between cluster and indices.
    D: int
        Embedded dimension.
    """
    N = data.shape[0]
    K = n_idx.shape[1]
    assert n_idx.shape == (N, K)

    # Intitial embeddings are randomly distributed.
    embeddings = np.random.normal(loc=0, scale=1, size=(N, D))

    return embeddings

def split_embeddings(embeddings, n_idx, divided_embeddings=None):
    """
    Split embeddings across neighborhoods.

    Parameters
    ----------
    embeddings: array-like
        Embeddings of data in D-d data.
    n_idx: list of list of ints
        Mapping between cluster and indices.
    divided_embeddings: array-like
        Replicated representation of embedded data.

    Returns
    -------
    divided_embeddings: array-like
        Embeddings replicated.
    """

    N, D = embeddings.shape
    K = n_idx.shape[1]
    assert n_idx.shape == (N, K)

    if divided_embeddings is None:
        divided_embeddings = np.zeros((K, N, D))
    assert divided_embeddings.shape == (K, N, D)

    for i, n_id in enumerate(n_idx):
        divided_embeddings[:, i, :] = embeddings[n_id]

    return divided_embeddings

def divide(divided_embeddings, scale_factors):
    """
    Maps a N x D embedding into K x N x D pivoted embeddings.
    The divide algorithm should take the neighborhood of each sample, move the
    sample to the centroid of the neighborhood, and move the points toward the
    centroid by a factor of tanh(||d||/sd - 1), where d is the distance to the
    centroid, and sd is the original distance.

    Parameters
    ----------
    divided_embeddings: array-like
        Replicated representation of embedded data.
    scale_factors: array-like
        Scale factor for each scaling, set initially by knn distances.

    Returns
    -------
    rval: array-like
        Replicated representation after pivoting.
    """

    rval = divided_embeddings.copy()
    K, N, D = rval.shape
    assert scale_factors.shape == (N, K)

    centroids = rval.mean(axis=0)

    # Set the center point to the mean and shift the data such that the centroid
    # is at the origin.
    rval[0] = centroids
    rval -= centroids

    # Compute the distances to the centers.
    distances = np.linalg.norm(rval,
                               axis=2)

    # Compute the scales as tanh(d/s - 1). Divide by 0 for source is fixed.
    scales = np.tanh(distances / scale_factors.T - 1)
    scales[0,:] = 0

    # Scale and shift back.  Broadcasting in numpy is trailing indices.
    rval = (scales.T * rval.T).T
    rval += centroids
    return rval

def concur(divided_embeddings, n_idx, embeddings=None):
    """
    Concur step.
    Sets all replications to the centroid.

    Parameters
    ----------
    divided_embeddings: array-like
        Replicated representation of embedded data.
    n_idx: list of list of ints
        Mapping between cluster and indices.
    embeddings: array-like
        Embeddings of data in D-d data.
    """
    N = divided_embeddings.shape[1]

    if embeddings is None:
        embeddings = np.zeros(divided_embeddings.shape[1:])
    assert embeddings.shape == divided_embeddings.shape[1:]

    for i in xrange(N):
        idx = np.where(n_idx.T == i)
        embeddings[i] = divided_embeddings[idx].mean(axis=0)
    return embeddings

def concur_and_split(divided_embeddings, n_idx, embeddings=None):
    """
    Concur then split.
    """
    embeddings = concur(divided_embeddings, n_idx, embeddings=None)
    return split_embeddings(embeddings, n_idx)

def d_and_c(data, K, D=2, beta=0.999, maxiter=100):
    """
    Divide and concur algorithm.
    Iterated though
    x_c = concur((1 + 1 / beta) * divide(x) - 1 / beta * x)
    and
    x_d = divide((1 - 1 / beta) * concur(x) + 1 / beta * x)
    finally:
    x <- beta(x_c - x_d)
    where x is the divided embedding.

    .. todo:: Add terminating criteria and monitoring.

    Parameters
    ----------
    data: array-like
        Input data.
    K: int
        k in knn.
    beta: float
        Learning rate.
    maxiter: int
        Maximum iterations of loop.

    Returns
    -------
    2d embeddings.
    """

    neighbors = knn(data, K)
    distances, n_idx = neighbors.kneighbors(data)
    embeddings = initialize_embeddings(data, n_idx, D=D)
    x = split_embeddings(embeddings, n_idx)

    for i in xrange(maxiter):
        stdout.write("\rIteration %d/%d                " % (i, maxiter))
        stdout.flush()
        #main loop
        x_c = concur_and_split(
            (1 + 1 / beta) * divide(x, distances) - 1 / beta * x,
            n_idx)
        x_d = divide((1 - 1 / beta) * concur_and_split(x, n_idx) + 1 / beta * x,
                     distances)
        x = x + beta * (x_c - x_d)

    return concur(x, n_idx)

def main(data, K, D=2):
    d_and_c(data, K, D=D)