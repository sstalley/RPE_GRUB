import networkx as nx
import grub












seed = 0x1359

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

        alg = grub.GRUB(g)

        print(f"graph:{g}, algorithm:{alg}")

        arm = alg.pick()

        alg.update(arm, 1.0)
