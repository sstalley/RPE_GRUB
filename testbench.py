import numpy as np
import networkx as nx
import grub

import matplotlib.pyplot as plt


# initialization
seed = 0x1359
np.random.seed(seed)
MAX_PULLS = 1000

#parameters from paper
n_nodes = [50, 100, 150, 200]
n_runs = 20
m = 2

# SBM model parameters
n_clusters = 10 # must divide all n_nodes values evenly
sbm_p = 0.95
sbm_q = 1e-4


np.set_printoptions(precision=3)


def run_sim(bandit, alg):
    while (not alg.done()) and alg.get_pulls() < MAX_PULLS:
        arm = alg.pick()
        reward = bandit.pull(arm)
        # print(f"Testbench: arm:{arm}, reward:{reward}")
        alg.update(arm, reward)

def test_graph(n, g, ba):

    grub_pulls = MAX_PULLS
    ucb_pulls = MAX_PULLS

    while (grub_pulls >= MAX_PULLS) or (ucb_pulls >= MAX_PULLS):
        # create bandit
        bandit = grub.Bandit(g)

        # calculate parameters
        smoothness = grub.calc_graph_smoothness(bandit.get_means(),g) / 4

        if ba:
            regularization = 1e-1
            subgaussian = 1e-1

        else:
            regularization = 5e-2
            subgaussian = 5e-2

        error_bound = 1e-2

        #create grub
        alg = grub.GRUB(g, regularization=regularization, smoothness=smoothness, subgaussian=subgaussian, error_bound=error_bound)

        #print(f"bandit:{bandit}, smoothness:{smoothness}, algorithm:{alg}")

        run_sim(bandit, alg)

        nc_graph = nx.empty_graph(n)
        ucb = grub.GRUB(nc_graph, regularization=regularization, smoothness=smoothness, subgaussian=subgaussian, error_bound=error_bound)

        run_sim(bandit, ucb)

        # make sure we didn't hit a runaway case
        grub_pulls = alg.get_pulls()
        ucb_pulls = ucb.get_pulls()



    true_mean = bandit.get_means()
    grub_mean = alg.get_means()
    ucb_mean = ucb.get_means()

    true_arm = np.argmax(true_mean)
    grub_arm = np.argmax(grub_mean)
    ucb_arm  = np.argmax(ucb_mean)

    debug = True
    if (debug):
        print(f"GRUB:     {alg.get_pulls()}")
        print(f"UCB:      {ucb.get_pulls()}")
        print(f"effective:{alg.get_effective_pulls():.3f}")

    if grub_arm != true_arm:
        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"GRUB best arm: {np.argmax(grub_mean)} with mean {np.max(grub_mean):.4f} after {alg.get_pulls()} pulls ({alg.get_effective_pulls()} effective)")

    if ucb_arm != true_arm:
        print(f"True best arm: {np.argmax(true_mean)} with mean {np.max(true_mean):.4f}")
        print(f"UCB  best arm: { np.argmax(ucb_mean)} with mean { np.max(ucb_mean):.4f} after {ucb.get_pulls()} pulls")

    return alg.get_pulls(), ucb.get_pulls(), alg.get_effective_pulls()


def run_simulations(ba=False):

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
            if (ba):
                g = nx.barabasi_albert_graph(n, m, seed=seed)
            else:
                choice = np.random.choice(range(n_clusters), size=n)
                #sizes = [len(choice[np.where(choice == i)]) for i in range(n_clusters)]
                sizes = [int(n/n_clusters) for i in range(n_clusters)] # know a-priori that there will be no remainder
                relation = (sbm_p - sbm_q) * np.eye(n_clusters) + sbm_q * np.ones((n_clusters, n_clusters))
                relation = (sbm_p - sbm_q) * np.eye(n_clusters) + sbm_q * np.ones((n_clusters, n_clusters))
                g = nx.stochastic_block_model(sizes, relation)

            grub_p, ucb_p, effective_p = test_graph(n, g, ba)
            grub_pull.append(grub_p)
            ucb_pull.append(ucb_p)
            grub_effective.append(effective_p)

        grub_pulls.append(np.mean(grub_pull))
        ucb_pulls.append(np.mean(ucb_pull))
        grub_effectives.append(np.mean(grub_effective))

    return np.array([grub_pulls, ucb_pulls, grub_effectives])



def plot_results(results, fig_name, title):
    grub_pulls = results[0]
    ucb_pulls = results[1]
    grub_effectives = results[2]


    plt.title(title)

    plt.scatter(n_nodes, grub_pulls,      c='r', marker='o', label="GRUB")
    plt.scatter(n_nodes, ucb_pulls,       c='b', marker='o', label="UCB")
    plt.scatter(n_nodes, grub_effectives, c='r', marker='x', label="GRUB (effective)")
    plt.legend()
    plt.ylabel("Pulls")
    plt.xlabel("Number of Arms")
    plt.savefig(fig_name, dpi=300)
    plt.clf()


sbm_cache_name = "./sbm_cache.npy"
sbm_fig_name = "sbm_figure.pdf"

ba_cache_name = "./ba_cache.npy"
ba_fig_name = "ba_figure.pdf"


try:
    print("trying to load cached values from", sbm_cache_name)
    sbm_results = np.load(sbm_cache_name)
    print("loaded cached values")

except IOError:
    print("could not load cache, running simulation")
    sbm_results = run_simulations()
    np.save(sbm_cache_name, sbm_results)

plot_results(sbm_results, sbm_fig_name, "Pulls per number of arms (Stochastic Block Model)")

try:
    print("trying to load cached values from", ba_cache_name)
    ba_results = np.load(ba_cache_name)
    print("loaded cached values")

except IOError:
    print("could not load cache, running simulation")
    ba_results = run_simulations(ba=True)
    np.save(ba_cache_name, ba_results)


plot_results(ba_results, ba_fig_name, "Pulls per number of arms (Barabasi-Albert Model)")

