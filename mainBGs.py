# -*- coding: utf-8 -*-
"""
"""

__author__ = 'marco'

import time
import os
import pickle
import numpy as np
from pathlib import Path

# NEST modules
import nest
MODULE_PATH = str(Path.home()) + '/nest/lib/nest/ml_module'
nest.Install(MODULE_PATH)       # Import my_BGs module

# my modules
from BGs_nest.BGs import BGs_class as B_c
from marco_nest_utils import utils, visualizer as vsl

PARAMS_DIR = 'BGs_nest/default_params.csv'  # contains model params

# simulation parameters
CORES = 4       # nest cores

cortical_mode = 'active'    # cortex is active or slow
in_vitro = False            # simulate neuron in vitro stimulation

dopa_depl_level = -0.8
if dopa_depl_level != 0.:
    dopa_depl = True
else:
    dopa_depl = False

N_tot_neurons = 20000       # number of network neurons
sim_time = 3000.0           # simulation time [ms]
start_time = 1000.           # starting time for histograms data

pop_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI', 'STN', 'SNr']

load_from_file = False

# set saving directory
savings_dir = f'savings/{cortical_mode}_{int(sim_time)}'
if in_vitro:
    savings_dir = savings_dir + '_invitro'
if dopa_depl:
    savings_dir = savings_dir + '_dopadepl'
print(f'Saving/Loading directories: {savings_dir}...')

TRIALS = 1
if __name__ == "__main__":
    out = []    # average firing rate buffer

    for i in range(TRIALS):
        savings_dir_trial = savings_dir + str(i)
        if not load_from_file:
            if not os.path.exists(savings_dir_trial): os.makedirs(savings_dir_trial)  # create folder if not present

        if not load_from_file:
            # set number of kernels
            nest.ResetKernel()
            nest.SetKernelStatus({'grng_seed': 100 * i + 1,
                                  'rng_seeds': [100 * i + 2, 100 * i + 3, 100 * i + 4, 100 * i + 5],
                                  'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
            nest.set_verbosity("M_ERROR")  # reduce plotted info

            # create an instance of the BGs populations and inputs
            BGs_class = B_c(nest, N_tot_neurons, cortical_mode, parameters_file_name=PARAMS_DIR,
                                  dopa_depl=dopa_depl_level, cortex_type='poisson', in_vitro=in_vitro)
            # a list of generated populations
            pop_list = [BGs_class.BGs_pops[name] for name in pop_names]

            # record membrane potential from the first neuron of the population
            vm_list = utils.attach_voltmeter(nest, pop_list, sampling_resolution=0.5, target_neurons=0)
            # record spikes from BGs neurons
            sd_list = utils.attach_spikedetector(nest, pop_list)

            # create a dictionary with model parameters
            model_dic = utils.create_model_dictionary(N_tot_neurons, pop_names, BGs_class.BGs_pop_ids, sim_time)

            tic = time.time()
            nest.Simulate(sim_time)
            toc = time.time()
            print(f'Elapsed simulation time with {CORES} cores: {int((toc - tic) / 60)} min, {(toc - tic) % 60:.0f} sec')

            potentials = utils.get_voltage_values(nest, vm_list, pop_names)
            rasters = utils.get_spike_values(nest, sd_list, pop_names)

            with open(f'{savings_dir_trial}/model_dic', 'wb') as pickle_file:
                pickle.dump(model_dic, pickle_file)
            with open(f'{savings_dir_trial}/potentials', 'wb') as pickle_file:
                pickle.dump(potentials, pickle_file)
            with open(f'{savings_dir_trial}/rasters', 'wb') as pickle_file:
                pickle.dump(rasters, pickle_file)

        elif load_from_file:
            print(f'Simulation results loaded from files')

            with open(f'{savings_dir_trial}/model_dic', 'rb') as pickle_file:
                model_dic = pickle.load(pickle_file)
            with open(f'{savings_dir_trial}/potentials', 'rb') as pickle_file:
                potentials = pickle.load(pickle_file)
            with open(f'{savings_dir_trial}/rasters', 'rb') as pickle_file:
                rasters = pickle.load(pickle_file)

        # show results
        fig1, ax1 = vsl.plot_potential_multiple(potentials, clms=1, t_start=start_time)
        fig1.show()

        fig2, ax2 = vsl.raster_plots_multiple(rasters, clms=1, start_stop_times=[0, sim_time], t_start=start_time)
        fig2.show()

        if not dopa_depl:
            ref_fr = {'slow': [0, 0, 0, 0, 0, 0, 26.0, 11.0, 0],
                      'active': [0, 0, 0, 0, 0, 0, 30.5, 12.0, 0]}
        else:
            ref_fr = {'slow': [0, 0, 0, 0, 9, 21.5, 0, 20.5, 0],
                      'active': [0, 0, 0, 0, 22, 15, 0, 30.0, 0]}
        fr_stats = utils.calculate_fr_stats(rasters, model_dic['pop_ids'], t_start=start_time)
        fig3, ax3 = vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['name'], CV_list=fr_stats['CV'],
                                              target_fr=ref_fr[cortical_mode])
        fig3.show()

        if i == 0:
            out = np.array([fr_stats['fr']])
        else:
            out = np.concatenate((out, np.array([fr_stats['fr']])), axis=0)

    mean_afr = out.mean(axis=0)
    std_afr = out.std(axis=0)

    print('mean_afr = ', mean_afr)
    print('std_afr = ', std_afr)
