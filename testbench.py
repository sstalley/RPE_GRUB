import numpy as np
import networkx as nx
import grub


# initialization
seed = 0x54f539
np.random.seed(seed)
MAX_PULLS = 50

# parameters for testing
n_nodes = [10]
n_runs = 1

#parameters from paper
#n_nodes = [50, 100, 150, 200]
#n_runs = 20
m = 2



for n in n_nodes:
    for run in range(n_runs):
        # Create Graph:
        g = nx.barabasi_albert_graph(n, m, seed=seed)

        bandit = grub.Bandit(g)
        alg = grub.GRUB(g)

        print(f"graph:{g}, algorithm:{alg}")

        while (not alg.done()) and bandit.get_pulls() < MAX_PULLS:

            arm = alg.pick()
            reward = bandit.pull(arm)
            alg.update(arm, reward)
