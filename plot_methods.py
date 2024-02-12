import propagation_methods as propag
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def extractStateCoordinate(states, i):
    return [state[i] for state in states]


def plot_signatures(signatures, regions, plot_bounds):
    
    x = extractStateCoordinate(signatures, 0)
    y = extractStateCoordinate(signatures, 1)

    dict_df = {'FirstCoordinate': x, 'SecondCoordinate': y}
    df = pd.DataFrame(dict_df)

    sns.set_palette("RdBu", 10)
    sns.set_style("ticks",{'axes.grid' : False})

    ax = sns.scatterplot(data=df, x='FirstCoordinate', y='SecondCoordinate')

    ax.set_xlim(plot_bounds[0], plot_bounds[1])
    ax.set_ylim(plot_bounds[0], plot_bounds[1])
    ax.set(xlabel='State[0]', ylabel='State[1]')

    for region in regions:
        ax.hlines(region[1], -15, 15, linewidth=0.5, color="gray")
        ax.vlines(region[0], -15, 15, linewidth=0.5, color="gray")



def plotSystemPropagation(initial_states, final_states, n_steps_ahead):
    
    initial_x0 = extractStateCoordinate(initial_states, 0)
    initial_x1 = extractStateCoordinate(initial_states, 1)

    final_x0 = extractStateCoordinate(final_states, 0)
    final_x1 = extractStateCoordinate(final_states, 1)

    dict_df = {'InitialStateX0': initial_x0, 'InitialStateX1': initial_x1, 'FinalStateX0': final_x0, 'FinalStateX1': final_x1}
    df = pd.DataFrame(dict_df)

    sns.set_palette("tab10")
    sns.set_style("ticks",{'axes.grid' : True})

    ax = sns.scatterplot(data=df, x='InitialStateX0', y='InitialStateX1')
    sns.scatterplot(data=df, x='FinalStateX0', y='FinalStateX1')
    
    plt.legend(labels=["Initial State", f"Final State (after {n_steps_ahead} steps)"])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set(xlabel='State[0]', ylabel='State[1]')


def plotSamplesFromGMM(samples):
        
    x = extractStateCoordinate(samples, 0)
    y = extractStateCoordinate(samples, 1)

    dict_df = {'FirstCoordinate': x, 'SecondCoordinate': y}
    df = pd.DataFrame(dict_df)

    sns.set_palette("Set2")
    sns.set_style("ticks",{'axes.grid' : True})

    ax = sns.scatterplot(data=df, x='FirstCoordinate', y='SecondCoordinate')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set(xlabel='State[0]', ylabel='State[1]')



def plotBounds(n_signatures, tv_bounds, wass_bounds):
    
    dict_df = {'NbOfSignatures': n_signatures, 'TVBounds': tv_bounds, 'WassBounds': wass_bounds}
    df = pd.DataFrame(dict_df)

    sns.set_palette("tab10")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    ax = sns.scatterplot(data=df, x='NbOfSignatures', y='TVBounds')
    sns.scatterplot(data=df, x='NbOfSignatures', y='WassBounds')
    
    plt.legend(labels=["TV bounds", "Wasserstein bounds"])

    ax.set(xlabel='Number of signatures for p(x)', ylabel='Distance bounds for p(y)')