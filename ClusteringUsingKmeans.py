# Importing required libraries
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Taking file path from system arguments
data_file = sys.argv[1]

# Reading data from text file into a dataframe
data = pd.read_csv(data_file, header=None, delim_whitespace=True)
# Dropping last column of data
data = data.drop(columns=data.columns[-1])

# Get series of data
def get_sample_value(x):
    return float(x.sample().iloc[0])

# Generating random centroids
def get_random_centroids(data, k):
    centroids = []
    np.random.seed(0)
    for i in range(k):
        centroid = data.apply(get_sample_value).reset_index(drop=True)
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

# Calculating euclidean distances
def euclidean_distances(x, centroid):
    return np.sqrt(((x - centroid) ** 2).sum(axis=1))

# Generating labels for unknown centroids
def get_labels(data, centroids):
    distances = pd.DataFrame(index=data.index, columns=centroids.columns)
    for centroid_col in centroids.columns:
        distances[centroid_col] = euclidean_distances(data, centroids[centroid_col])
    return distances.idxmin(axis=1)

# Calculating geometric mean
def geometric_mean(x):
    small_constant = 1e-10
    return np.exp(np.log(x + small_constant).mean())

# Updating centroids usinng geometric mean
def get_new_centroids(data, labels, k):
    grouped_data = data.groupby(labels)
    centroids = grouped_data.apply(geometric_mean).T
    return centroids

# Calculating error
def calculate_error(data, labels, centroids):
    distances = pd.DataFrame(index=data.index, columns=centroids.columns)
    for centroid_col in centroids.columns:
        distances[centroid_col] = euclidean_distances(data, centroids[centroid_col])
    return np.sum(distances.min(axis=1))

# K-means clustering algorithm
def kmeans(data, k, max_iterations=20):
    centroids = get_random_centroids(data, k)
    for _ in range(max_iterations):
        labels = get_labels(data, centroids)
        new_centroids = get_new_centroids(data, labels, k)
        if centroids.equals(new_centroids):
            break
        centroids = new_centroids
    error = calculate_error(data, labels, centroids)
    return error

# Plotting Error vs K chart
k_values = range(2, 11)
errors = []

for k in k_values:
    error = kmeans(data, k)
    errors.append(error)
    print(f"For k = {k} After 20 iterations: Error = {error}")

plt.plot(k_values, errors, marker='o')
plt.title('Error vs k chart')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Error')
plt.show()