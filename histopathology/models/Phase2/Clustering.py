import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform
from pprint import pprint
from tqdm import tqdm


def xie_beni_index(embeddings, labels, cluster_centers):
    n_clusters = len(np.unique(labels))
    min_dist = np.min(
        [np.linalg.norm(c1 - c2) for i, c1 in enumerate(cluster_centers) for c2 in cluster_centers[i + 1:]])
    intra_cluster_dist = np.sum(
        [np.linalg.norm(embeddings[labels == i] - cluster_centers[i], axis=1).sum() for i in range(n_clusters)])
    return intra_cluster_dist / (embeddings.shape[0] * min_dist)


def c_index(embeddings, labels):
    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    max_dist = np.max(dist_matrix)
    intra_dist = np.sum([np.sum(dist_matrix[labels == i][:, labels == i]) for i in np.unique(labels)]) / 2
    c = (intra_dist - min_dist) / (max_dist - min_dist)
    return c


def hartigan_index(embeddings, labels, cluster_centers):
    n_clusters = len(np.unique(labels))
    intra_cluster_dist = np.sum(
        [np.linalg.norm(embeddings[labels == i] - cluster_centers[i], axis=1).sum() for i in range(n_clusters)])
    total_variance = np.var(embeddings, axis=0).sum() * embeddings.shape[0]
    return intra_cluster_dist / total_variance


def dunn_index(embeddings, labels, cluster_centers):
    n_clusters = len(np.unique(labels))
    intra_cluster_distances = [
        np.max([np.linalg.norm(x1 - x2) for x1 in embeddings[labels == i] for x2 in embeddings[labels == i]]) for i in
        range(n_clusters)]
    inter_cluster_distances = [np.linalg.norm(c1 - c2) for i, c1 in enumerate(cluster_centers) for c2 in
                               cluster_centers[i + 1:]]
    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


def mclain_rao_index(embeddings, labels):
    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))
    inter_cluster_dists = np.sum([np.min(dist_matrix[labels == i][:, labels != i]) for i in np.unique(labels)])
    intra_cluster_dists = np.sum([np.sum(dist_matrix[labels == i][:, labels == i]) for i in np.unique(labels)])
    return inter_cluster_dists / intra_cluster_dists


def evaluate_cluster(embeddings, labels, true_labels):
    cluster_centers = np.array([embeddings[labels == i].mean(axis=0) for i in np.unique(labels)])
    xie_beni = xie_beni_index(embeddings, labels, cluster_centers)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    # c = c_index(embeddings, labels)
    hartigan = hartigan_index(embeddings, labels, cluster_centers)
    # dunn = dunn_index(embeddings, labels, cluster_centers)
    # mclain_rao = mclain_rao_index(embeddings, labels)

    return {
        'Xie-Beni Index': xie_beni,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin,
        # 'C Index': c,
        'Hartigan Index': hartigan,
        # 'Dunn Index': dunn,
        # 'McLain-Rao Index': mclain_rao
    }


def run_clustering(embeddings, true_labels):
    results = {}
    cluster_range = range(3, 10)
    clustering_algorithms = [
        ('KMeans', MiniBatchKMeans),
        ('BisectingKMeans', BisectingKMeans),
        ('GaussianMixture', GaussianMixture)
    ]
    for name, algorithm in tqdm(clustering_algorithms, desc='Clustering Algorithms'):
        for n_clusters in tqdm(cluster_range, desc=f'{name} Clusters', leave=False):
            if name == 'GaussianMixture':
                model = algorithm(n_components=n_clusters, random_state=420)
                labels = model.fit_predict(embeddings)
            else:
                model = algorithm(n_clusters=n_clusters, random_state=420)
                labels = model.fit_predict(embeddings)
            results[f'{name}_{n_clusters}'] = evaluate_cluster(embeddings, labels, true_labels)
    return results


dae_embeddings = pd.read_csv("../../data/processed/DAE_Embeddings.csv").iloc[:, :-1]
true_label = pd.read_csv("../../data/processed/DAE_Embeddings.csv").iloc[:, -1]
print("Data Loading Finished, Running Clustering")
results = run_clustering(dae_embeddings, true_label)

# Display results
# import ace_tools as tools; tools.display_dataframe_to_user(name="Clustering Evaluation Results", dataframe=pd.DataFrame(results))

pprint(results)

# Optionally, save the results to a CSV file
pd.DataFrame(results).to_csv('clustering_evaluation_results.csv', index=False)
