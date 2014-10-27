#!/usr/bin/env python

import divide_and_concur as dc
import numpy as np

def test_knn(dim=7, samples=70):
    """
    Function to test basic KNN functionality.
    """

    data = np.zeros((samples, dim))
    # Normal distribution, each centered around s/10,
    # where s is the sample number
    for s in xrange(samples):
        data[s] = np.random.normal(loc=(s//10), scale=0.1, size=dim)
    neighbors = dc.knn(data, K=samples//10)

    # With a tight normal distribution as above, neighbors should be in
    # subsets of 10s
    for k in xrange(samples//10):
        distances, n_idx = neighbors.kneighbors(data[k * 10])
        assert set(n_idx.tolist()[0]).issubset(
            set(range(k * 10, (k + 1) * 10))),\
            "Unexpected set: %r and %r" % (set(n_idx.tolist()[0]),
                                           set(range(k * 10, (k + 1) * 10)))

    return data, neighbors

def test_initialization():
    data, neighbors = test_knn()
    distances, n_idx = neighbors.kneighbors(data)
    embeddings = dc.initialize_embeddings(data, n_idx)
    # Divide embeddings per neighborhood.
    divided_embeddings = split_embeddings(embeddings, n_idx)

    for i, n_id in enumerate(n_idx):
        assert np.all(divided_embeddings[:, i, :] == embeddings[n_id])

def test_divide():
    data,  = test_knn()
    distances, n_idx = neighbors.kneighbors(data)
    embeddings = dc.initialize_embeddings(data, n_idx)
    # Divide embeddings per neighborhood.
    divided_embeddings = split_embeddings(embeddings, n_idx)

    # Sadly there is muting and copying in divide.
    divide_before = divided_embeddings.copy()
    divide_after = dc.divide(divided_embeddings,
                             distances)
    assert np.allclose(divide_after[0],
                       divide_before.mean(axis=0)),\
        "Means not close:\nCentroid:\n%r\nMean:\n%r" %\
        (divide_after[0], divide_before.mean(axis=0))

    distances_before = np.linalg.norm(
        divide_before - divide_before.mean(axis=0), axis=2)
    distances_after = np.linalg.norm(
        divide_after - divide_after[0], axis=2)
    assert np.all(distances_before > distances_after)

def test_concur():
    data, neighbors = test_knn()
    distances, n_idx = neighbors.kneighbors(data)
    embeddings = dc.initialize_embeddings(data, n_idx)
    # Divide embeddings per neighborhood.
    divided_embeddings = split_embeddings(embeddings, n_idx)

    dc.concur(divided_embeddings, n_idx)
