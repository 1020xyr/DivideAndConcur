#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN


def knn(data, K):
    neighbors = KNN(n_neighbors=K)
    neighbors.fit(data, data)
    return neighbors

def initialize_embeddings(data, n_idx, D=2):
    """
    Function to initialize embeddings and broadcast them into a divided one.
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
    """

    K, N, D = divided_embeddings.shape
    assert scale_factors.shape == (N, K)

    centroids = divided_embeddings.mean(axis=0)

    # Set the center point to the mean and shift the data such that the centroid
    # is at the origin.
    divided_embeddings[0] = centroids
    divided_embeddings -= centroids

    # Compute the distances to the centers.
    distances = np.linalg.norm(divided_embeddings,
                               axis=2)

    # Compute the scales as tanh(d/s - 1). Divide by 0 for source is fixed.
    scales = np.tanh(distances / scale_factors.T - 1)
    scales[0,:] = 0

    # Scale and shift back.  Broadcasting in numpy is trailing indices.
    divided_embeddings = (scales.T * divided_embeddings.T).T
    divided_embeddings += centroids
    return divided_embeddings

def concur(divided_embeddings, n_idx, embeddings=None):
    N = divided_embeddings.shape[1]

    if embeddings is None:
        embeddings = np.zeros(divided_embeddings.shape[1:])
    assert embeddings.shape == divided_embeddings.shape[1:]

    for i in xrange(N):
        idx = np.where(n_idx.T == i)
        embeddings[i] = divided_embeddings[idx].mean(axis=0)
    return embeddings

def d_and_c(data, neighbors, D=2, beta=0.999, maxiter=100):
    distances, n_idx = neighbors.kneighbors(data)
    embeddings = dc.initialize_embeddings(data, n_idx)

    divided_embeddings = None
    for i in xrange(maxiter):
        #main loop
        pass

def main(data, K):
    neighbors = knn(data, K)
