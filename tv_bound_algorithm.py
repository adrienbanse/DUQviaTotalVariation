import torch
import parameters
import grid_generation as grid
import total_variation_bound as tv
import probability_mass_computation as proba

from distributions import GaussianMixture



unbounded_region = torch.Tensor([[torch.inf, torch.inf], [torch.inf, torch.inf]])  # a representation choice for the unbounded region

def tv_bound_algorithm(dynamics, initial_distribution, noise_distribution, barrier):

    tv_bounds = [0.0]
    gmms = []


    for t in range(parameters.n_steps_ahead + 1):

        if t == 0:
            hat_gmm = initial_distribution
        else:
            means_gmm = dynamics(signatures)
            covs_noise = noise_distribution.covariance.unsqueeze(0).expand(means_gmm.size(0), -1, -1)
            hat_gmm = GaussianMixture(means_gmm, covs_noise, double_hat_probs)

        gmms.append(hat_gmm)
        samples = hat_gmm(parameters.n_samples)

        high_prob_region = grid.identify_high_prob_region(samples)
        outer_signature = grid.outer_point(high_prob_region)

        regions = grid.create_regions(high_prob_region, samples, parameters.min_proportion, parameters.min_size)
        signatures = grid.place_signatures(regions)

        double_hat_probs = proba.compute_signature_probabilities(hat_gmm.means, hat_gmm.covariances[0], hat_gmm.weights, regions) #TODO: Generalize for GMMs with different covariances

        regions, signatures = grid.add_unbounded_representations(regions, unbounded_region, signatures, outer_signature)

        tv_bound, contributions = tv.compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, double_hat_probs, regions)


        for r in range(parameters.n_refinements):

            regions = regions[:-1]
            signatures = signatures[:-1]
            regions, signatures = grid.refine_regions(regions, signatures, contributions, parameters.threshold)

            double_hat_probs = proba.compute_signature_probabilities(hat_gmm.means, hat_gmm.covariances[0], hat_gmm.weights, regions) #TODO: Generalize for GMMs with different covariances
            regions, signatures = grid.add_unbounded_representations(regions, unbounded_region, signatures, outer_signature)

            tv_bound, contributions = tv.compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, double_hat_probs, regions)


        tv_bounds.append(tv_bound)

    return tv_bounds, gmms