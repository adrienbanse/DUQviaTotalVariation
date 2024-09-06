#Time horizon
n_steps_ahead = 5

#Monte Carlo simulation
n_samples = 10000

#Grid parameters
grid_type = "uniform_grid" #if 'uniform_grid', set n_refinements to zero
min_proportion = 0.01
min_size = 0.001
max_depth = 3 #only used if 'uniform_grid' is chosen

#Refinement
threshold = 1e-7
n_refinements = 0

#Plot
colors = ['Purples', 'Blues', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']