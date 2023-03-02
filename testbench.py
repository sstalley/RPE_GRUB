import numpy as np
import networkx as nx
import grub


# initialization
seed = 0x54f539
np.random.seed(seed)
MAX_PULLS = 500

# parameters for testing
n_nodes = [10]
n_runs = 1

#parameters from paper
#n_nodes = [50, 100, 150, 200]
#n_runs = 20
m = 2


np.set_printoptions(precision=3)

for n in n_nodes:
    for run in range(n_runs):
        # Create Graph:
        g = nx.barabasi_albert_graph(n, m, seed=seed)

        # create bandit
        bandit = grub.Bandit(g)

        # calculate parameters
        smoothness = grub.calc_graph_smoothness(bandit.get_means(),g)

        #create grub
        alg = grub.GRUB(g, smoothness=smoothness)

        print(f"bandit:{bandit}, smoothness:{smoothness}, algorithm:{alg}")

        while (not alg.done()) and bandit.get_pulls() < MAX_PULLS:

            arm = alg.pick()
            reward = bandit.pull(arm)

            print(f"Testbench: arm:{arm}, reward:{reward}")

            alg.update(arm, reward)

        true_mean = bandit.get_means()
        est_mean = alg.get_means()

        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"Estimated best arm: {np.argmax(est_mean)} with mean {np.max(est_mean):.4f}")

        print(f"Total estimation error: {np.linalg.norm(est_mean-true_mean):.4f}")
