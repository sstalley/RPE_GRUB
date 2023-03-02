import numpy as np
import networkx as nx
import grub


# initialization
seed = 0x54f539
np.random.seed(seed)

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

        mean_lin = np.random.multivariate_normal(np.zeros(n), 2*np.eye(n))
        smooth_lin = grub.calc_graph_smoothness(mean_lin, g)
        print(f"linear means:{mean_lin}")
        print(f"linear mean smoothness:{smooth_lin}")


        cov = np.abs(np.array(nx.normalized_laplacian_matrix(g).toarray()))
        print(f"covariance matrix:{cov}")

        mean_cov = np.random.multivariate_normal(np.zeros(n), cov)
        smooth_cov = grub.calc_graph_smoothness(mean_cov, g)
        print(f"correlated means:{mean_cov}")
        print(f"correlated mean smoothness:{smooth_cov}")

        alg = grub.GRUB(g)

        print(f"graph:{g}, algorithm:{alg}")

        arm = alg.pick()

        alg.update(arm, 1.0)
