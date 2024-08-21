#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:00:43 EDT 2024

@author: Andrius Burnelis
"""
import numpy as np, warnings
import sys; sys.path.append('./')
from data_loader import DataLoader
import models, os, tqdm, time
import emcee, ptemcee
from scipy.stats import truncnorm

warnings.simplefilter('ignore')



def main():
    ##############################################################################
    ##############################################################################
    # Code to change between runs
    ##############################################################################
    ##############################################################################
    # # Select the data subsets to use
    # # E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    # # E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    E_min = 0.676 # MeV
    E_max = 4.342 # MeV
    which_data = 'som'

    # Select the parameterizations
    parameterization = 'bs_C'

    # Parameters for the MCMC sampling 
    # Multiply this by the number of walkers to get the total number of samples obtained after running
    n_steps = 300
    n_burns = 100

    # Use ptemcee?
    use_ptemcee = True

    # Parameters to set the number of different temperatures (only for ptemcee)
    n_temps_low = 4
    n_temps_high = 4

    # Use theory cov?
    use_theory_cov = True

    # Always write a comment for the run (at least just '\n'!!)
    comment = 'Long run 3.a, 4.3 MeV, introduced c_bar^2 prior cap at 15\n'

    # Set the save_name (without the file extension - the extension depends on the sampler)
    save_name = 'samples'

    # # # Optional:
    # Set a specific prior? (Default set to None)
    params_bound = None
    params_prior = None
    ##############################################################################
    ##############################################################################
    # End code to change between runs
    ##############################################################################
    ##############################################################################


    # # # Load in the data
    loader = DataLoader(E_min, E_max, which_data)
    data = loader.get_data()
    norm_group = loader.get_normalization_grouping()
    gauss_prior_f = loader.get_normalization_prior_info()


    # # # Set the parameter bounds and initialize the model
    if parameterization == 'bs_C':
        if param_bounds is None:
            param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]]) # Default values: [lower, upper]
        if params_prior is None:
            params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # Default values: [center, width]
        gauss_prior_params = np.hstack([param_bounds, params_prior])
        model = models.BS_C(data, norm_group, gauss_prior_params, gauss_prior_f, use_theory_cov)
    else:
        sys.stderr.write('Nothing else is implemented yet...')
        sys.exit(-1)

    # Send the model to the proper sampler
    if use_ptemcee:
        run_ptemcee(model, n_burns, n_steps, n_temps_low, n_temps_high, save_name, parameterization)
    else:
        run_emcee(model, n_burns, n_steps, save_name)


def run_emcee(model, n_burn : int, n_step : int, save_name : str):
    # Set the save name
    save_name = save_name + '.h5'

    # Initialize the starting samples (according to the prior)
    starting_samples = []
    for j in range(0, model.total_dim):
        min_bound, max_bound, mu, sigma = model.prior_info[j]
        lower = (min_bound - mu) / sigma
        upper = (max_bound - mu) / sigma
        starting_samples.append(truncnorm.rvs(lower, upper, loc = mu, scale = sigma, size = (2 * model.total_dim)))

    # Cast to an array
    starting_samples = np.column_stack(starting_samples)

    # # # Set up the backend
    n_walkers = int(model.total_dim * 2)
    backend = emcee.backends.HDFBackend(save_name)
    backend.reset(n_walkers, model.total_dim)

    sys.stdout.write('******************** Getting Samples (burn-in: {}) {} ********************\n'.format(n_burn, time.ctime()))

    # # # Initialize the emcee ensemble sampler (Without multiprocessing)
    sampler = emcee.EnsembleSampler(n_walkers, model.total_dim, 
                model.log_posterior, 
                # moves = [(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(), 0.5)],
                backend = backend)
    
    # # # Run the burnin and sample
    # Execute the burn in
    pos, prob, state = sampler.run_mcmc(starting_samples, n_burn, progress = True)
    sys.stdout.write('******************** Getting Samples (steps: {}) {} ********************\n'.format(n_step, time.ctime()))

    # Run the sampler for n_step
    running = sampler.run_mcmc(pos, n_step, progress = True)

    sys.stdout.write('Finished run at {}\n'.format(time.ctime()))



def run_ptemcee(model, n_burn : int, n_step : int, n_temps_low : int, n_temps_high : int, save_name : str, parameterization : str = 'bs_C'):
    # Set the save name
    save_name = save_name + '.npz'
    
    # # # # Set up the MCMC parameters (walkers, betas, etc...)
    n_walkers = int(2 * model.total_dim)
    temps_low = np.array([2**(i / 8) for i in range(0, n_temps_low)])
    temps_high = np.array([np.sqrt(2)**i for i in range(0, n_temps_high)])
    temps = np.concatenate((temps_low, temps_high[temps_high > max(temps_low)]))
    n_temps = temps.shape[0]
    betas = 1 / temps

    if parameterization == 'bs_C':
        # Initialize the starting samples (according to the prior)
        starting_samples = np.zeros((n_temps, n_walkers, model.total_dim))
        for k in range(0, n_temps):
            for j in range(0, n_walkers):
                for l in range(0, model.total_dim):
                    min_bound, max_bound, mu, sigma = model.prior_info[l]
                    lower = (min_bound - mu) / sigma
                    upper = (max_bound - mu) / sigma
                    starting_samples[k, j, l] = truncnorm.rvs(lower, upper, loc = mu, scale = sigma)
    else:
         sys.stdout.write('Parameterization not recognized\n')
         sys.exit(-1)

    # Cast to an array
    # starting_samples = np.column_stack(starting_samples)

    # # # # Initialize the sampler
    sampler = ptemcee.Sampler(n_walkers, model.total_dim, model.log_likelihood, model.log_prior, n_temps, betas = betas)


    # # # # Run the burn-in and sample
    count = 0
    sys.stdout.write('******************** Getting Samples (burn-in: {}) {} ********************\n'.format(n_burn, time.ctime()))
    for p, lnprob, lnlike in sampler.sample(starting_samples, iterations = n_burn):
        count += 1
        bar_len = 60
        filled_len = int(round(bar_len * count / float(n_burn)))
        percents = round(100.0 * count / float(n_burn), 1)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', ''))
        sys.stdout.flush()

    sampler.reset()

    sys.stdout.write('Finished burn in - Starting run at {}\n'.format(time.ctime()))

    count = 0
    sys.stdout.write('******************** Getting Samples (steps: {}) {} ********************\n'.format(n_step, time.ctime()))
    for p, lnprob, lnlike in sampler.sample(p, iterations = n_step):
        count += 1
        bar_len = 60
        filled_len = int(round(bar_len * count / float(n_step)))
        percents = round(100.0 * count / float(n_step), 1)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', ''))
        sys.stdout.flush()
    sys.stdout.write('\n')

    # Save the samples
    pt_samples = sampler.chain
    np.savez(save_name, pt_samples)

    sys.stdout.write('Finished run at {}\n'.format(time.ctime()))

    sys.stdout.write('Saved run to {}\n'.format(save_name))























































































































































































































































































































# def main():
#     ##############################################################################
#     ##############################################################################
#     # Code to change between runs
#     ##############################################################################
#     ##############################################################################
#     # # Select the data subsets to use
#     # # E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
#     # # E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
#     E_mins = np.array([0.676]) # MeV
#     E_maxes = np.array([4.342]) # MeV
#     which_datas = ['som']

#     # Select the parameterizations
#     parameterizations = ['bs_C']

#     # Parameters for the MCMC sampling 
#     # Multiply by the number of walkers!!!
#     n_steps = [30000]
#     n_burns = [10000]

#     # Parameters to set the number of different temperatures
#     n_temps_lows = [4]
#     n_temps_highs = [4]

#     # Use theory cov?
#     use_theory_covs = [True]

#     # Always write a comment for the run (at least just '\n'!!)
#     comments = ['\n']

#     # # # Optional:
#     # Set a specific prior? (Default set to None)
#     params_bounds = [None]
#     params_priors = [None]

#     # params_bounds = [np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [1.70, 3], [5.0, 25.0], [-6, 6]])]
#     # params_priors = [np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [2.0, 1.6], [12.59, 1.85], [0.0, 1.6]])]
#     ##############################################################################
#     ##############################################################################
#     # End code to change between runs
#     ##############################################################################
#     ##############################################################################



#     ##############################################################################
#     # # # Create the directory structure to keep track of samples

#     # Get the time and format the parent directory
#     current_time = time.strftime("%m_%d_%Y")
#     parent_directory = './' + current_time

#     # Create the parent directory with os and get the max starting number
#     if not os.path.exists(parent_directory):
#         os.mkdir(parent_directory)
#         start_count = 0
#     else:
#         current_sub_dirs = os.listdir(parent_directory)
#         start_count = max([int(''.join(filter(str.isdigit, x))) for x in current_sub_dirs]) + 1

#     # # Generate a list of subdirectories
#     subdirectories = ["run{}".format(i + start_count) for i in range(0, E_mins.shape[0])]

#     # Create the subdirectories with os
#     for subdirectory in subdirectories:
#         if not os.path.exists(parent_directory + '/' + subdirectory):
#             os.mkdir(parent_directory + '/' + subdirectory)

#     # Generate the list of save names
#     save_names = ["{}/{}/samples_{}.npz".format(parent_directory, subdirectories[i], i + start_count) for i in range(0, E_mins.shape[0])]
#     ##############################################################################


#     ##############################################################################
#     # # # Do a run for each element in the user defined lists
#     for i in range(0, E_maxes.shape[0]):
#         # Set the run variables
#         E_min = E_mins[i]
#         E_max = E_maxes[i]
#         which_data = which_datas[i]
#         parameterization = parameterizations[i]
#         use_theory_cov = use_theory_covs[i]
#         n_step = n_steps[i]
#         n_burn = n_burns[i]
#         comment = comments[i]
#         save_name = save_names[i]
#         n_temps_low = n_temps_lows[i]
#         n_temps_high = n_temps_highs[i]

#         # Handle if we do not specify priors
#         try:
#             param_bounds = params_bounds[i]
#             params_prior = params_priors[i]
#         except IndexError as e:
#             param_bounds = None
#             params_prior = None

#         # # # Load in the data
#         loader = DataLoader(E_min, E_max, which_data)
#         data = loader.get_data()
#         norm_group = loader.get_normalization_grouping()
#         gauss_prior_f = loader.get_normalization_prior_info()

#         # # # Set up the READ_ME
#         with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', '+w') as f:
#             f.write("Choice of model: {}\n".format(parameterization))
#             f.write("Number of temps: {} low, {} high\n".format(n_temps_low, n_temps_high))
#             f.write("Analyzing {} data with E_min {} MeV and E_max {} MeV\n".format(which_data, E_min, E_max))
#             f.write("Theory covariance: {}\n\nCustom Prior Bounds: \n{}\n\nCustom Prior Params: \n{}\n\n".format(use_theory_cov, param_bounds, params_prior))

#         # # # Set the parameter bounds and initialize the model
#         if parameterization == 'sim_bs_C':
#             if param_bounds is None:
#                 param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
#             if params_prior is None:
#                 params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
#             gauss_prior_params = np.hstack([param_bounds, params_prior])
#             model = models.Sim_BS_C(data, norm_group, gauss_prior_params, gauss_prior_f, use_theory_cov)
#         else:
#             sys.stderr.write('Nothing else is implemented yet...')
#             sys.exit(-1)

#         # # # # Set up the MCMC parameters (walkers, betas, etc...)
#         n_walkers = int(2 * model.total_dim)
#         temps_low = np.array([2**(i / 8) for i in range(0, n_temps_low)])
#         temps_high = np.array([np.sqrt(2)**i for i in range(0, n_temps_high)])
#         temps = np.concatenate((temps_low, temps_high[temps_high > max(temps_low)]))
#         # temps = np.linspace(1, 100, int(n_temps_low + n_temps_high))
#         n_temps = temps.shape[0]
#         betas = 1 / temps

#         # # # Continue with the README
#         with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
#             f.write("Run with {} steps, {} burn in and {} walkers.\n".format(n_step, n_burn, n_walkers))
#             f.write("Custom comment: \n{}\n".format(comment))

#         ##############################################################################

#         # # # Initialize the starting samples (according to the prior)
#         # For the ERPs + norms
#         starting_samples = np.zeros((n_temps, n_walkers, model.total_dim))
#         for k in range(0, n_temps):
#             for j in range(0, n_walkers):
#                 for l in range(2, model.total_dim):
#                     min_bound, max_bound, mu, sigma = model.prior_info[l - 2]
#                     lower = (min_bound - mu) / sigma
#                     upper = (max_bound - mu) / sigma
#                     starting_samples[k, j, l] = truncnorm.rvs(lower, upper, loc = mu, scale = sigma)

   
#         # Useful output statements
#         sys.stdout.write('Starting run with {} data {} - {} MeV\n'.format(which_data, E_min, E_max))
#         sys.stdout.write('MCMC sampling using emcee (affine invariant ensamble sampler) with {} walkers and {} steps\n'.format(n_walkers, n_step))
#         N = data.shape[0]
#         sys.stdout.write('The number of input data points are: {}\nThe number of parameters are: {}\n'.format(N, model.total_dim))
#         # sys.stdout.write('Sampling will be split across {} cores.\n'.format(cpu_use))
#         # # # Write useful information to the README
#         with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
#             f.write('\n\nThe number of input data points are: {}\nThe number of parameters are: {}\n'.format(N, model.total_dim))
#             f.write('Started burn in at {}\n'.format(time.ctime()))
        
#         # # # # Initialize the sampler
#         sampler = ptemcee.Sampler(n_walkers, model.total_dim, model.log_likelihood, model.log_prior, n_temps,
#                                 #   threads = cpu_use, 
#                                   betas = betas)

#         # # # # Run the burn-in and sample
#         count = 0
#         sys.stdout.write('******************** Getting Samples (burn-in: {}) {} ********************\n'.format(n_burn, time.ctime()))
#         for p, lnprob, lnlike in sampler.sample(starting_samples, iterations = n_burn):
#             count += 1
#             bar_len = 60
#             filled_len = int(round(bar_len * count / float(n_burn)))
#             percents = round(100.0 * count / float(n_burn), 1)
#             bar = '█' * filled_len + '-' * (bar_len - filled_len)
#             sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', ''))
#             sys.stdout.flush()

#         sampler.reset()

#         with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
#             f.write('Finished burn in - Starting run at {}\n'.format(time.ctime()))

#         count = 0
#         sys.stdout.write('******************** Getting Samples (steps: {}) {} ********************\n'.format(n_step, time.ctime()))
#         for p, lnprob, lnlike in sampler.sample(p, iterations = n_step):
#             count += 1
#             bar_len = 60
#             filled_len = int(round(bar_len * count / float(n_step)))
#             percents = round(100.0 * count / float(n_step), 1)
#             bar = '█' * filled_len + '-' * (bar_len - filled_len)
#             sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', ''))
#             sys.stdout.flush()
#         sys.stdout.write('\n')

#         with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
#             f.write('Finished run at {}\n'.format(time.ctime()))

#         # Save the samples
#         pt_samples = sampler.chain
#         np.savez(save_name, pt_samples)

#         sys.stdout.write('Saved run to {}\n'.format(save_name))





if __name__ == "__main__":
    main()