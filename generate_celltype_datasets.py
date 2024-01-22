import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import hnn_core
from hnn_core import calcium_model, simulate_dipole, read_params, pick_connection
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.network_builder import NetworkBuilder
from hnn_core.cell import _get_gaussian_connection
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils
from utils import (SingleNeuron_Data, Network_Data, CellType_Dataset_Fast,
                   linear_scale_forward, log_scale_forward, UniformPrior, section_drive_param_function)
import multiprocessing
from joblib import Parallel, delayed
n_sims = 1000
# device = torch.device("cuda:0")
device = 'cpu'

# num_cores = multiprocessing.cpu_count()
num_cores = 32


# Define simulation function
#---------------------------
def run_hnn(thetai, sample_idx, prior_dict, transform_dict=None, suffix='subthreshold', rate=20):

    data_path = f'/users/ntolley/scratch/bayesian_surrogates/datasets_{suffix}'

    theta_dict = {param_name: param_dict['rescale_function'](thetai[param_idx].numpy(), param_dict['bounds']) for 
                    param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}

    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3, 'N_pyr_y': 3})
    
    net = calcium_model(params)
    if suffix != 'connected':
        net.clear_connectivity()

    theta_extra = {'cell_type_lookup': cell_type_lookup, 'valid_conn_list': list(),
                   'valid_drive_dict': dict()}
    # Connections
    for conn_idx in range(len(net.connectivity)):
        src_type = cell_type_lookup[net.connectivity[conn_idx]['src_type']]
        target_type = cell_type_lookup[net.connectivity[conn_idx]['target_type']]
        receptor = net.connectivity[conn_idx]['receptor']
        loc = net.connectivity[conn_idx]['loc']
        
        conn_name = f'{src_type}_{target_type}_{receptor}_{loc}'
        theta_extra['valid_conn_list'].append(conn_name)
        theta_extra[f'{conn_name}_conn_idx'] = conn_idx

    # Drives
    for cell_type in net.cell_types.keys():
        for sec_name in net.cell_types[cell_type].sections.keys():
            for syn_name in net.cell_types[cell_type].sections[sec_name].syns:
                drive_name = f'{cell_type}_{sec_name}_{syn_name}'
                theta_extra['valid_drive_dict'][drive_name] = {
                    'cell_type': cell_type, 'location': sec_name, 'receptor': syn_name}

    theta_extra['sample_idx'] =  sample_idx
    theta_dict['theta_extra'] = theta_extra

    section_drive_param_function(net, theta_dict, rate=rate)
    dpl = simulate_dipole(net, dt=0.5, tstop=500, record_vsec='all', record_isec='all', record_dcell=True)

    # g = net.cell_response.plot_spikes_raster(show=False)
    # g.savefig(f'{data_path}/raster_plots/raster_{sample_idx}.png')
    # plt.close()

    # g = dpl[0].plot(show=False)
    # g.savefig(f'{data_path}/dipole_plots/dipole_{sample_idx}.png')
    # plt.close()

    # g = dpl[0].plot_psd(fmin=0, fmax=100, show=False)
    # g.savefig(f'{data_path}/psd_plots/psd_{sample_idx}.png')
    # plt.close()

    # np.save(f'{data_path}/dipole_data/dipole_{sample_idx}.npy', dpl[0].data['agg'], )

    for cell_type in net.cell_types.keys():
        if transform_dict is None:
            input_spike_scaler, vsec_scaler, isec_scaler = None, None, None
        else:
            input_spike_scaler = transform_dict[cell_type]['input_spike_scaler']
            vsec_scaler = transform_dict[cell_type]['vsec_scaler']
            isec_scaler = transform_dict[cell_type]['isec_scaler']

        training_set = utils.CellType_Dataset_Fast(
            net, cell_type=cell_type, window_size=1000, data_step_size=1000,
            input_spike_scaler=input_spike_scaler, vsec_scaler=vsec_scaler, isec_scaler=isec_scaler,
            soma_filter=False, device='cpu')
        torch.save(training_set, f'{data_path}/training_data/{cell_type}_dataset_{sample_idx}.pt')


# Generate subthreshold dataset
#------------------------------
suffix = 'suprathreshold'
rate = 10.0
net = calcium_model()

cell_type_lookup = {
    'L2_pyramidal': 'L2e', 'L2_basket': 'L2i',
    'L5_pyramidal': 'L5e', 'L5_basket': 'L5i'}


prior_dict = dict()
prior_dict['lamtha'] = {'bounds': (0, 10), 'rescale_function': linear_scale_forward}

# Connections
for conn_idx in range(len(net.connectivity)):
    src_type = cell_type_lookup[net.connectivity[conn_idx]['src_type']]
    target_type = cell_type_lookup[net.connectivity[conn_idx]['target_type']]
    receptor = net.connectivity[conn_idx]['receptor']
    loc = net.connectivity[conn_idx]['loc']
    
    conn_name = f'{src_type}_{target_type}_{receptor}_{loc}'
    
    prior_dict[f'{conn_name}_gbar'] = {'bounds': (-4, 0), 'rescale_function': log_scale_forward}
    prior_dict[f'{conn_name}_prob'] = {'bounds': (0, 1), 'rescale_function': linear_scale_forward}

# Drives
for cell_type in net.cell_types.keys():
    for sec_name in net.cell_types[cell_type].sections.keys():
        for syn_name in net.cell_types[cell_type].sections[sec_name].syns:
            drive_name = f'{cell_type}_{sec_name}_{syn_name}'
            prior_dict[f'{drive_name}_gbar'] = {'bounds': (-4,-0), 'rescale_function': log_scale_forward}
            prior_dict[f'{drive_name}_prob'] = {'bounds': (0, 1), 'rescale_function': linear_scale_forward}


prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((n_sims,))

# First sample used to fit transformers
theta_samples[0,:] = torch.from_numpy(np.repeat(0.7, theta_samples.shape[1]))
run_hnn(theta_samples[0, :], 0, prior_dict, transform_dict=None, suffix=suffix, rate=rate)

transform_dict = {}
for cell_type in net.cell_types.keys():
    dataset = torch.load(f'/users/ntolley/scratch/bayesian_surrogates/datasets_subthreshold/training_data/{cell_type}_dataset_0.pt')
    transform_dict[cell_type] = {'input_spike_scaler': dataset.input_spike_scaler,
                                 'vsec_scaler': dataset.vsec_scaler,
                                 'isec_scaler': dataset.isec_scaler}
    
# Skip first sample which is used for creating transforms
Parallel(n_jobs=num_cores)(delayed(run_hnn)(
    thetai, sample_idx+1, prior_dict, transform_dict, suffix, rate) for
    (sample_idx, thetai) in enumerate(theta_samples[1:, :]))



# Generate suprathreshold dataset
#------------------------------
suffix = 'subthreshold'
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((n_sims,))
rate = 10.0

# Drives
for cell_type in net.cell_types.keys():
    for sec_name in net.cell_types[cell_type].sections.keys():
        for syn_name in net.cell_types[cell_type].sections[sec_name].syns:
            drive_name = f'{cell_type}_{sec_name}_{syn_name}'
            prior_dict[f'{drive_name}_gbar'] = {'bounds': (-4, -3), 'rescale_function': log_scale_forward}
            prior_dict[f'{drive_name}_prob'] = {'bounds': (0, 1), 'rescale_function': linear_scale_forward}

# update_keys = ['L2e_distal', 'L2i_distal', 'L5e_distal', 'L5i_distal',
#                'L2e_proximal', 'L2i_proximal', 'L5e_proximal', 'L5i_proximal']
# for key in update_keys:
#     prior_dict[key]['bounds'] = (lower_g, upper_g)

Parallel(n_jobs=num_cores)(delayed(run_hnn)(
    thetai, sample_idx, prior_dict, transform_dict, suffix, rate) for
    (sample_idx, thetai) in enumerate(theta_samples))


# # Generate connected dataset
# #---------------------------
# suffix = 'connected'
# prior = UniformPrior(parameters=list(prior_dict.keys()))
# theta_samples = prior.sample((n_sims,))

# Parallel(n_jobs=num_cores)(delayed(run_hnn)(
#     thetai, sample_idx, prior_dict, transform_dict, suffix, rate) for
#     (sample_idx, thetai) in enumerate(theta_samples))


