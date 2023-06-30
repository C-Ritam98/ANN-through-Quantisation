import numpy as np
import random
import copy

class Quantisation:

    def __init__(self, Input_vector_set: np.ndarray, compression_factor : int, number_of_clusters : int):
        
        self.Input_vector_set = Input_vector_set
        X = copy.deepcopy(Input_vector_set)
        N = X.shape[0] # number of vectors
        DIM = X.shape[1] # embedding dimension of the individual vectors

        assert compression_factor > 1, "[ERROR] compression_factor should be mrore than 1"
        assert DIM%compression_factor == 0, "[ERROR] compression_factor should divide DIM evenly !"
        assert number_of_clusters <= 256, "[ERROR] number_of_clusters should be an integer less than 256 such that it can be stored in 1 Byte !"

        self.num_subspaces = DIM / compression_factor

        self.dim_subspace = compression_factor

        self.number_of_clusters = number_of_clusters

        self.SPLITS = np.split(X, self.num_subspaces, axis = 1)

        self.K_means = self.Compute_K_means_of_all_subspaces()

        self.quantised_vectors =  self.quantise_across_all_subspaces(self.K_means, self.SPLITS)


    def help(self):
        help_ = '''
        Currently only supports compression through Product Quantisation ! \n
        Input Parameters:
        compression_factor: Input Embedding size / Embedding size of the compressed vector
        number_of_clusters: The 'K' in k-means. Should be less than 256 to store the id in 1 Byte. Typically K = min(256, 10* Num_data_points)
        '''
        print(help_)

    def Compute_K_means_of_all_subspaces(self):
        print("[INFO] Computing K-means clusters !!")

        K_means = np.array([self.Compute_K_means_given_subspace(split, self.number_of_clusters) for split in self.SPLITS])

        print("[INFO] K-means clusters computation done !!")
        return K_means
    

    def Compute_K_means_given_subspace(self, split : np.ndarray, number_of_clusters: int):

        print("[INFO] Computing K-means sub-clusters !!")

        eps = 1e-4

        k_means_idx = np.random.choice(split.shape[0], number_of_clusters, replace=False) # random initialisation of means from the given set of vectors
        k_means = split[k_means_idx] # fetching the means
        old_k_means = copy.deepcopy(k_means)

        while True:
            
            closest_mean_ids = np.argmin(np.sum((k_means - split.reshape(-1, 1, self.dim_subspace)) ** 2, axis = 2), axis = 1)
            
            for id in range(self.number_of_clusters):
                X = split[np.where(closest_mean_ids == id)[0]]
                k_means[id, :] = np.mean(X, axis = 0)

            if np.sqrt(np.sum((old_k_means - k_means) ** 2)) < 1.:
                break

            old_k_means = copy.deepcopy(k_means)

        # print("[INFO] K-means clusters computation done !!")
        return k_means


    def quantise_across_all_subspaces(self, K_means : np.ndarray, SPLITS : np.ndarray):
        assert len(K_means) == len(SPLITS)
        quantised_ids = []
        for i in range(len(K_means)):
            quantised_ids.append(self.quantise(K_means[i], SPLITS[i]))

        quantised_ids = np.array(quantised_ids)

        return quantised_ids.T # shape = N x num_sub_spaces


    def quantise(self, K_means, Split):

        TEMP = K_means - Split.reshape(-1, 1, self.dim_subspace)

        Dist = np.sum(TEMP ** 2, axis = 2)

        closest_cluster_mean_id = np.argmin(Dist, axis = 1)

        return closest_cluster_mean_id
    

    def get_quantised_vectors(self, query_vectors: np.array):
        
        assert query_vectors.shape[1] == self.dim_subspace * self.num_subspaces, 'Query vector(s) not in same embedding space as the database vectors !'

        N = query_vectors.shape[0]

        X = np.split(query_vectors, self.num_subspaces, axis = 1)

        return self.quantise_across_all_subspaces(self.K_means, X)
    
    def hamming_distance(self, query, vectors):
        distances = np.sum(1 - (vectors == query), axis = 1)
        ids = np.argsort(distances)
        return self.Input_vector_set[ids], ids

    def get_approximate_nearest_neighbors(self, query: np.ndarray, top_k: int):
        
        quantised_query = self.get_quantised_vectors(query.reshape(1, -1))
        # print(quantised_query, quantised_query.shape)

        vectors_with_closest_hamming_distance, ids = self.hamming_distance(quantised_query, self.quantised_vectors)
        vectors_with_closest_hamming_distance, ids = vectors_with_closest_hamming_distance[:top_k*10], ids[:top_k*10]
        vectors_with_closest_hamming_distance = vectors_with_closest_hamming_distance / np.linalg.norm(vectors_with_closest_hamming_distance, axis = 1).reshape(-1,1)
        query = query / np.linalg.norm(query)

        dot = query.reshape(1,-1) @ vectors_with_closest_hamming_distance.T
        ids = ids[np.argsort(-dot.ravel())]

        return ids[1:top_k+1]
