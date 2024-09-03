import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import barriers as barriers

colors = ['Purples', 'Blues', 'Oranges', 'YlOrBr', 'YlOrRd',
          'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu',
          'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

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

        plt.hist2d(states[:, 0], states[:, 1], bins=100, cmap=colors[t], alpha=0.8, cmin=0.1)

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