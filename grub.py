import numpy as np
import networkx as nx

class GRUB():


    #guessing on smoothness
    def __init__(self, g, regularization=2.0, smoothness=0.1, error_bound=1e-3, subgaussian=2.0):
        #determine number of arms from graph
        self.n_arms = nx.number_of_nodes(g)
        self.V = regularization * np.array(nx.laplacian_matrix(g).toarray())
        self.n_pulls = 0
        self.n_components = nx.number_connected_components(g)

        self.regularization = regularization
        self.smoothness = smoothness
        self.error_bound = error_bound
        self.subgaussian = subgaussian


    def __str__(self):
        return f"GRUB: {self.n_arms} arms, {self.n_pulls} pulls"
