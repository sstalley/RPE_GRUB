import numpy as np
import networkx as nx
import grub

import matplotlib.pyplot as plt


# initialization
seed = 0x1359
np.random.seed(seed)
MAX_PULLS = 5000

#parameters from paper
n_nodes = [50, 100, 150, 200]
n_runs = 20
m = 2


np.set_printoptions(precision=3)


def run_sim(bandit, alg):
    while (not alg.done()) and alg.get_pulls() < MAX_PULLS:
        arm = alg.pick()
        reward = bandit.pull(arm)
        # print(f"Testbench: arm:{arm}, reward:{reward}")
        alg.update(arm, reward)

def test_graph(n, g):

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
    grub_mean = alg.get_means()
    ucb_mean = ucb.get_means()

    true_arm = np.argmax(true_mean)
    grub_arm = np.argmax(grub_mean)
    ucb_arm  = np.argmax(ucb_mean)

    if grub_arm != true_arm:
        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"GRUB best arm: {np.argmax(grub_mean)} with mean {np.max(grub_mean):.4f} after {alg.get_pulls()} pulls ({alg.get_effective_pulls()} effective)")

    if ucb_arm != true_arm:
        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"UCB  best arm: { np.argmax(ucb_mean)} with mean { np.max(ucb_mean):.4f} after {ucb.get_pulls()} pulls")

    return alg.get_pulls(), ucb.get_pulls(), alg.get_effective_pulls()


def run_simulations():

    grub_pulls = []
    ucb_pulls  = []
    grub_effectives = []

    for n in n_nodes:
        grub_pull = []
        ucb_pull = []
        grub_effective = []
        for run in range(n_runs):
            print(f"{n} arm run #{run}")

            # Create Graph:
            g = nx.barabasi_albert_graph(n, m, seed=seed)

            grub_pull, ucb_pull, grub_effective = test_graph(n, g)


        grub_pulls.append(np.mean(grub_pull))
        ucb_pulls.append(np.mean(ucb_pull))
        grub_effectives.append(np.mean(grub_effective))

    return np.array([grub_pulls, ucb_pulls, grub_effectives])



cache_name = "./sbm_cache.npy"
fig_name = "BA_figure.eps"

try:
    print("trying to load cached values from", cache_name)
    results = np.load(cache_name)
    print("loaded cached values")

except IOError:
    print("could not load cache, running simulation")
    results = run_simulations()
    np.save(cache_name, results)

print(results)

grub_pulls = results[0]
ucb_pulls = results[1]
grub_effectives = results[2]


plt.title("Average Pulls per Number of Arms")

plt.scatter(n_nodes, grub_pulls, label="GRUB")
plt.scatter(n_nodes, ucb_pulls, label="UCB")
plt.scatter(n_nodes, grub_effectives, label="GRUB (effective)")
plt.legend()
plt.savefig(fig_name, dpi=300)

