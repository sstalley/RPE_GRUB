import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

def calc_graph_smoothness(mean, g):
    laplacian = np.array(nx.laplacian_matrix(g).toarray())
    return np.sqrt(mean.T @ laplacian @ mean)

GEN_TIMEOUT = 1000

class Bandit():

    def __init__(self, g, sd=0.05, scale=20, min_prominence=1.0):
        self.n_pulls = 0
        self.n_arms = nx.number_of_nodes(g)
        self.g = g
        cov = np.abs(np.array(nx.normalized_laplacian_matrix(g).toarray()))
        self.means = np.random.multivariate_normal(np.zeros(self.n_arms), scale * cov)
        self.sd = sd

        timeout = 0
        while((self.prominence() < min_prominence) and timeout < GEN_TIMEOUT):
            self.means = np.random.multivariate_normal(np.zeros(self.n_arms), scale * cov)

        # print(f"prominence: {self.prominence()}")

    def pull(self, arm):
        noise = 0
        self.n_pulls = self.n_pulls + 1

        if self.sd > 0:
            noise = np.random.normal(scale=self.sd)

        return self.means[arm] + noise

    def prominence(self):
        sorted_means = np.sort(self.means)
        return sorted_means[-1] - sorted_means[-2]


    def get_means(self):
        return self.means


    def __str__(self):
        return f"Bandit: {self.n_arms} arms, {self.n_pulls} pulls"

class GRUB():

    def _calc_beta(self):
        return self.regularization * self.smoothness + 2 * self.subgaussian * np.sqrt(14*np.log( 2*self.n_arms*self.teff*self.teff / self.error_bound))

    def _model_ready(self):
        return self.n_pulls >= self.n_components

    def _select_least_effective_pulls(self):

        arms = np.argsort(self.teff)

        for arm in arms:
            # only pick arms that are still good
            if self.good_arms[arm]:
                # print(f"selecting arm {arm} with {self.teff[arm]:.3f} effective pulls")
                return arm

    def __init__(self, g, regularization=0.1, smoothness=0.1, error_bound=1e-1, subgaussian=1e-2, sampling_policy="min_teff"):
        #determine number of arms from graph
        self.n_arms = nx.number_of_nodes(g)
        self.V = regularization * np.array(nx.laplacian_matrix(g).toarray())
        self.x = np.zeros((self.n_arms))
        self.n_pulls = 0
        self.components = nx.connected_components(g)
        self.n_components = nx.number_connected_components(g)

        #initally all good
        self.good_arms = np.full((self.n_arms), True)

        if sampling_policy == "min_teff":
            self.sampling_policy = self._select_least_effective_pulls

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
            # print(f"GRUB: picking arm {arm} from component {self.n_pulls}")
            self.n_pulls = self.n_pulls + 1
            return arm

        #select an arm
        arm = self.sampling_policy()
        self.n_pulls = self.n_pulls + 1
        return arm


    def update(self, arm, reward, update_model=True):

        # print(f"GRUB: updating arm {arm} with reward {reward:.3f}...")
        self.V[arm,arm] = self.V[arm,arm] + 1
        self.x[arm] = self.x[arm] + reward

        if update_model and self._model_ready():
            # print(f"GRUB: updating model...")
            self.V_inv = np.linalg.inv(self.V)
            self.mean = self.V_inv @ self.x
            self.teff = 1 / np.diagonal(self.V_inv)
            self.beta = self._calc_beta()

            # Calculate bounding
            self.bound = np.sqrt(1/self.teff) * self.beta
            lower_bound = self.mean - self.bound
            upper_bound = self.mean + self.bound
            # print(f"bound:{self.bound}")

            # find best arm (highest lower bound)
            best_arm = np.argmax(self.mean - self.bound)
            best_lb = lower_bound[best_arm]

            # print(f"GRUB: current best arm:{best_arm} with lower bound {best_lb:.3f}")
            # compare with other arms
            # print(f"GRUB: current upper bounds:{upper_bound}")

            self.good_arms = upper_bound > best_lb

            # print(f"total pulls:{self.n_pulls}, effective pulls:{np.sum(self.teff):.3f}")

            # plt.imshow(self.V_inv, interpolation='nearest')
            # plt.show()


    def done(self):

        if not self._model_ready():
            return False

        lower_bound = self.mean - self.bound
        upper_bound = self.mean + self.bound

        # find best arm (highest lower bound)
        best_arm = np.argmax(self.mean - self.bound)
        best_lb = lower_bound[best_arm]

        remain = np.count_nonzero(self.good_arms)

        # print(f"GRUB: {remain} arms still under consideration")

        if remain == 1:
            return True

        return False

    def get_means(self):
        return self.mean

    def get_pulls(self):
        return self.n_pulls

    def get_effective_pulls(self):
        if self._model_ready():
            return np.sum(self.teff)
        else:
            return self.n_pulls

