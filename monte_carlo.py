import parameters
import barriers as barriers
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import probability_mass_computation as proba

def monte_carlo_simulation(dynamics, initial_distribution, noise_distribution, barrier, n_simulations, n_samples):

    fig, ax = plt.subplots()
    hitting_probas = []

    rect = Rectangle(barrier[0], barrier[1][0] - barrier[0][0], barrier[1][1] - barrier[0][1],
                     edgecolor='red', facecolor='lightcoral', fill=True, lw=1, alpha=0.7, label='Unsafe set')
    ax.add_patch(rect)

    states = initial_distribution(n_samples)

    for t in range(n_simulations + 1):

        if t > 0:
            states = dynamics(states)
            states = states + noise_distribution(n_samples) #additive noise

        #Plot
        plt.hist2d(states[:, 0], states[:, 1], bins=100, alpha=0.8, cmin=0.1)

        hitting_proba = barriers.hitting_probability(states, barrier)
        hitting_probas.append(hitting_proba)

    plt.legend(loc='lower right')
    plt.xlim(1, 9)
    plt.ylim(0, 11)
    plt.xlabel('State[0]')
    plt.ylabel('State[1]')
    plt.grid(True)
    plt.show()

    return hitting_probas

def gmm_approximation_monte_carlo(gmms, barrier, n_samples):

    gmm_hitting_probs = []

    fig, ax = plt.subplots()

    rect = Rectangle(barrier[0], barrier[1][0] - barrier[0][0], barrier[1][1] - barrier[0][1],
                     edgecolor='red', facecolor='lightcoral', fill=True, lw=1, alpha=0.7, label='Unsafe set')
    ax.add_patch(rect)

    for gmm in gmms:

        # Compute hitting probability
        proba_barrier = proba.gaussian_mixture_proba_mass_inside_hypercubes(gmm.means, gmm.covariances[0], gmm.weights, barrier.unsqueeze(0))
        gmm_hitting_probs.append(proba_barrier.item())

        samples = gmm(n_samples)

        #Plot
        plt.hist2d(samples[:, 0], samples[:, 1], bins=100, alpha=0.8, cmin=0.1)


    plt.legend(loc='lower right')

    plt.xlim(1, 9)
    plt.ylim(0, 11)
    plt.xlabel('State[0]')
    plt.ylabel('State[1]')
    plt.grid(True)
    plt.show()

    return gmm_hitting_probs
