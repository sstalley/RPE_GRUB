import numpy as np
import networkx as nx
import grub


# initialization
seed = 0x1359
np.random.seed(seed)
MAX_PULLS = 5000

# parameters for testing
n_nodes = [50, 100, 150, 200, 250]
n_runs = 20

#parameters from paper
#n_nodes = [50, 100, 150, 200]
#n_runs = 20
m = 2


np.set_printoptions(precision=3)


def run_sim(bandit, alg):
    while (not alg.done()) and alg.get_pulls() < MAX_PULLS:
        arm = alg.pick()
        reward = bandit.pull(arm)
        # print(f"Testbench: arm:{arm}, reward:{reward}")
        alg.update(arm, reward)



for n in n_nodes:
    for run in range(n_runs):
        # Create Graph:
        g = nx.barabasi_albert_graph(n, m, seed=seed)

        # create bandit
        bandit = grub.Bandit(g)

        # calculate parameters
        smoothness = grub.calc_graph_smoothness(bandit.get_means(),g) / 4

        regularization = 1e-1
        subgaussian = 1e-1
        error_bound = 1e-2

        #create grub
        alg = grub.GRUB(g, regularization=regularization, smoothness=smoothness, subgaussian=subgaussian, error_bound=error_bound)

        #print(f"bandit:{bandit}, smoothness:{smoothness}, algorithm:{alg}")

        run_sim(bandit, alg)

        nc_graph = nx.empty_graph(n)
        ucb = grub.GRUB(nc_graph, regularization=regularization, smoothness=smoothness, subgaussian=subgaussian, error_bound=error_bound)

        run_sim(bandit, ucb)

        true_mean = bandit.get_means()
        est_mean = alg.get_means()
        ucb_mean = ucb.get_means()

        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"GRUB best arm: {np.argmax(est_mean)} with mean {np.max(est_mean):.4f} after {alg.get_pulls()} pulls ({alg.get_effective_pulls()} effective)")
        print(f"UCB  best arm: {np.argmax(ucb_mean)} with mean {np.max(ucb_mean):.4f} after {ucb.get_pulls()} pulls ({ucb.get_effective_pulls()} effective)")
        # print(f"Total estimation error: {np.linalg.norm(est_mean-true_mean):.4f}")
