import numpy as np

class KMeans:

    def __init__(self, k = 3, tolerance = 0.01, max_iter = 100, runs = 1,
                 init_method="forgy"):
        """Clustering algorithm that generates groups

        Args:
            k (int, optional): Number of clusters. Defaults to 3.
            tolerance (float, optional): It is a value that must no be exceeded
                by the difference betwen the distances of the centroids of the
                current and previous iteration. Defaults to 0.01.
            max_iter (int, optional): Maximum number of iterations.
                Defaults to 100.
            runs (int, optional): Number of times the algorithm will be execute.
                Defaults to 1.
            init_method (str, optional): Method to inicialize centroids.
                Defaults to "forgy".
        """
        self.k = k
        self.tolerance = tolerance
        self.cluster_means = np.zeros(k)
        self.max_iter = max_iter
        self.init_method = init_method

        # There is no need to run the algorithm multiple times if the
        # initialization method is not a random process
        self.runs = runs if init_method == 'forgy' else 1

    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)

    def __compute_cost(self, X, labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis = 1).sum()

        return cost

    def __initialize_means(self, X, row_count):
        if self.init_method == 'forgy':
            return forgy(X, row_count, self.k)
        elif self.init_method == 'maximin':
            return min_max(X, self.k)
        elif self.init_method == 'macqueen':
            return macqueen(X, self.k)
        elif self.init_method == 'var_part':
            return var_part(X, self.k)
        else:
            raise Exception('The initialization method {} does not exist or not implemented'.format(self.init_method))

    def __compute_distances(self, X, cluster_means, row_count):
        distances = np.zeros((row_count, self.k))
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis = 1)

        return distances

    def __label_examples(self, distances):
        return distances.argmin(axis = 1)

    def __compute_means(self, X, labels, col_count):
        cluster_means = np.zeros((self.k, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            if len(cluster_elements):
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis = 0)

        return cluster_means

    def fit(self, data):
        rows, columns = data.shape

        values = self.__get_values(data)

        labels = np.zeros(rows)

        costs = np.zeros(self.runs)
        all_clusterings = list()

        for i in range(self.runs):
            cluster_means =  self.__initialize_means(values, rows)

            for _ in range(self.max_iter):
                previous_means = np.copy(cluster_means)

                distances = self.__compute_distances(values, cluster_means, rows)

                labels = self.__label_examples(distances)

                cluster_means = self.__compute_means(values, labels, columns)

                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break

            values_with_labels = np.append(values, labels[:, np.newaxis], axis = 1)

            all_clusterings.append( (cluster_means, values_with_labels) )
            costs[i] = self.__compute_cost(values, labels, cluster_means)

        best_clustering_index = costs.argmin()

        self.cost_ = costs[best_clustering_index]

        return all_clusterings[best_clustering_index]


def forgy(data, rows, k):
    """
        **Forgy Method**

        Randomly select k observations from the data set.

        :param data: data set
        :type data: pandas.core.frame.DataFrame
        :param rows: Number of rows in the dataset
        :type rows: int
        :param k: Number of clusters
        :type k: int
    """
    return data[np.random.choice(rows, replace=False, size=k)]

def macqueen(data, k):
    """
        **Macqueen Method**

        K centroids are taken that will later be recalculated

        :param data: data set
        :type data: pandas.core.frame.DataFrame
        :param k: Number of clusters
        :type k: int
    """
    return data[:k]

def min_max(data, k):
    """
        **MinMax Method**

        Maximizes the distance between the centroids of the clusters

        :param data: data set
        :type data: pandas.core.frame.DataFrame
        :param k: Number of clusters
        :type k: int
    """
    copy_data = np.copy(data)
    # Se crea una matriz con los valores RGB para cada centroide
    initial_centers = np.zeros((k, copy_data.shape[1]))
    # Retorna la norma de la matriz
    copy_datanorms = np.linalg.norm(copy_data, axis = 1)
    # retorna el índice del valores máximo
    copy_datanorms_max_i = copy_datanorms.argmax()
    # Se ubica el primer centroide
    initial_centers[0] = copy_data[copy_datanorms_max_i]
    # Se elimina el valor máximo del data set
    copy_data = np.delete(copy_data, copy_datanorms_max_i, axis = 0)
    for i in range(1, k):
        distances = np.zeros((copy_data.shape[0], i))
        for index, center in enumerate(initial_centers[:i]):
            distances[:, index] = np.linalg.norm(copy_data - center, axis = 1)

        max_min_index = distances.min(axis = 1).argmax()

        initial_centers[i] = copy_data[max_min_index]
        copy_data = np.delete(copy_data, max_min_index, axis = 0)

    return initial_centers

def var_part(data, k):
    # ! Completar
    data_ = np.append(data, np.zeros(data.shape[0])[:, np.newaxis], axis = 1)
    initial_centers = np.zeros((k, data.shape[1]))

    cluster_i = 1
    while cluster_i != k:
        withik_sum_squares = np.zeros(cluster_i)
        for j in range(cluster_i):
            cluster_members = data_[ data_[:, -1] == j ]
            cluster_mean = cluster_members.mean(axis = 0)
            withik_sum_squares[j] = np.linalg.norm(cluster_members - cluster_mean, axis = 1).sum()

        # Cluster which has greatest SSE
        max_sse_i = withik_sum_squares.argmax()
        data_max_sse_i = data_[:, -1] == max_sse_i
        data_max_sse = data_ [ data_max_sse_i ]

        variances, means = data_max_sse.var(axis = 0), data_max_sse.mean(axis = 0)
        max_variance_i = variances.argmax()
        max_variance_mean = means [ max_variance_i ]

        data_smaller_mean = data_max_sse[:, max_variance_i] <= max_variance_mean
        data_greater_mean = data_max_sse[:, max_variance_i] > max_variance_mean

        initial_centers[max_sse_i] = data_max_sse [ data_smaller_mean ].mean(axis = 0)[:-1]
        initial_centers[cluster_i] = data_max_sse [ data_greater_mean ].mean(axis = 0)[:-1]

        data_[ (data_max_sse_i) & (data_ [:, max_variance_i] <= max_variance_mean), -1] = cluster_i
        data_[ (data_max_sse_i) & (data_ [:, max_variance_i] > max_variance_mean), -1] = max_sse_i

        cluster_i += 1

    return initial_centers