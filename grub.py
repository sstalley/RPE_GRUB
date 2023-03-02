import numpy as np
import networkx as nx


def calc_graph_smoothness(mean, g):
    laplacian = np.array(nx.laplacian_matrix(g).toarray())
    return np.sqrt(mean.T @ laplacian @ mean)


class GRUB():

    def _model_ready(self):
        return self.n_pulls >= self.n_components

    #guessing on smoothness
    def __init__(self, g, regularization=2.0, smoothness=0.1, error_bound=1e-3, subgaussian=2.0):
        #determine number of arms from graph
        self.n_arms = nx.number_of_nodes(g)
        self.V = regularization * np.array(nx.laplacian_matrix(g).toarray())
        self.x = np.zeros((self.n_arms))
        self.n_pulls = 0
        self.components = nx.connected_components(g)
        self.n_components = nx.number_connected_components(g)


        self.regularization = regularization
        self.smoothness = smoothness
        self.error_bound = error_bound
        self.subgaussian = subgaussian


    def __str__(self):
        return f"GRUB: {self.n_arms} arms, {self.n_pulls} pulls"


    def pick(self):

        # sample every graph component first
        if not self._model_ready():
            arm = np.random.choice(list(next(self.components)))

            print(f"arm:{arm}")

            print(f"GRUB: picking arm {arm} from component {self.n_pulls}")
            self.n_pulls += 1
            return arm


    def update(self, arm, reward, update_model=True):

        print(f"GRUB: updating arm {arm} with reward {reward}...")
        self.V[arm,arm] =+ 1
        self.x[arm] =+ reward

        if update_model and self._model_ready():
            print(f"GRUB: updating model...")
            self.V_inv = np.linalg.inv(self.V)
            self.mean = self.V_inv @ self.x
            print(f"x shape:{self.x.shape}, mean shape:{self.mean.shape}")

