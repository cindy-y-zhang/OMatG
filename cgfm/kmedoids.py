"""
Periodic k-medoids clustering of a single crystal structure.

This is the fixed, purely geometric partition that the learned grouping has to beat in order for the experiment to say
anything. Medoids rather than means are used because the cluster centre must be an actual atom under the minimum-image
metric, where averaging positions is not well defined.

The implementation is deliberately plain: an exact pairwise distance matrix, k-medoids++ seeding and alternating
assignment and medoid updates. Structures here have at most 52 atoms, so nothing about this needs to be clever, and a
transparent implementation is worth more than a fast one for a partition that is computed once and cached.
"""

import numpy as np


def min_image_distance_matrix(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Compute the matrix of minimum-image Cartesian distances between the atoms of one structure.

    :param frac:
        Fractional coordinates of shape (N, 3).
    :type frac: numpy.ndarray
    :param cell:
        Cell vectors of shape (3, 3), with row i the i-th lattice vector.
    :type cell: numpy.ndarray

    :return:
        Symmetric distance matrix of shape (N, N) with a zero diagonal.
    :rtype: numpy.ndarray
    """
    difference = frac[None, :, :] - frac[:, None, :]
    difference = difference - np.round(difference)
    return np.linalg.norm(difference @ cell, axis=-1)


def _seed_medoids(distances: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Choose initial medoids with k-medoids++ seeding.

    :param distances:
        Symmetric distance matrix of shape (N, N).
    :type distances: numpy.ndarray
    :param num_clusters:
        Number of clusters K.
    :type num_clusters: int
    :param rng:
        Random generator, seeded by the caller for reproducibility.
    :type rng: numpy.random.Generator

    :return:
        Indices of the initial medoids, of shape (K,).
    :rtype: numpy.ndarray
    """
    num_atoms = distances.shape[0]
    medoids = [int(rng.integers(num_atoms))]
    while len(medoids) < num_clusters:
        nearest = distances[:, medoids].min(axis=1)
        nearest[medoids] = 0.0
        total = nearest.sum()
        if total <= 0.0:
            # All remaining atoms coincide with a medoid; fall back to any unused atom.
            remaining = [i for i in range(num_atoms) if i not in medoids]
            medoids.append(int(remaining[0]))
        else:
            medoids.append(int(rng.choice(num_atoms, p=nearest / total)))
    return np.array(medoids, dtype=np.int64)


def periodic_kmedoids(distances: np.ndarray, num_clusters: int, seed: int = 0,
                      max_iterations: int = 100) -> np.ndarray:
    """
    Partition the atoms of one structure into k clusters by alternating k-medoids.

    :param distances:
        Symmetric matrix of minimum-image distances of shape (N, N).
    :type distances: numpy.ndarray
    :param num_clusters:
        Number of clusters K, at most the number of atoms.
    :type num_clusters: int
    :param seed:
        Seed of the random generator used for k-medoids++ seeding.
        Defaults to 0.
    :type seed: int
    :param max_iterations:
        Largest number of assignment and update rounds.
        Defaults to 100.
    :type max_iterations: int

    :return:
        Cluster label of every atom, taking every value in [0, K) exactly once per medoid, of shape (N,).
    :rtype: numpy.ndarray

    :raises ValueError:
        If the number of clusters is not between one and the number of atoms.
    """
    num_atoms = distances.shape[0]
    if not 1 <= num_clusters <= num_atoms:
        raise ValueError(f"Cannot form {num_clusters} clusters from {num_atoms} atoms.")

    rng = np.random.default_rng(seed)
    medoids = _seed_medoids(distances, num_clusters, rng)
    labels = np.zeros(num_atoms, dtype=np.int64)
    for _ in range(max_iterations):
        new_labels = np.argmin(distances[:, medoids], axis=1)
        # Every medoid keeps its own cluster, which guarantees that no cluster becomes empty even when several atoms
        # sit at identical distances from several medoids.
        new_labels[medoids] = np.arange(num_clusters)
        new_medoids = medoids.copy()
        for cluster in range(num_clusters):
            members = np.flatnonzero(new_labels == cluster)
            new_medoids[cluster] = members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))]
        if np.array_equal(new_labels, labels) and np.array_equal(new_medoids, medoids):
            break
        labels, medoids = new_labels, new_medoids
    return labels
