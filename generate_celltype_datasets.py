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
                   linear_scale_forward, log_scale_forward, UniformPrior, beta_tuning_param_function)
import multiprocessing
from joblib import Parallel, delayed
n_sims = 100
# device = torch.device("cuda:0")
device = 'cpu'

num_cores = multiprocessing.cpu_count()


# Define simulation function
#---------------------------
def run_hnn(thetai, sample_idx, prior_dict, transform_dict=None, suffix='subthreshold'):
    theta_dict = {param_name: param_dict['rescale_function'](thetai[param_idx].numpy(), param_dict['bounds']) for 
                    param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}

    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3, 'N_pyr_y': 3})
    
    net = calcium_model(params)
    if suffix != 'connected':
        net.clear_connectivity()

    # Extract all E-I connection types
    E_gids = np.concatenate([net.gid_ranges['L2_pyramidal'], net.gid_ranges['L5_pyramidal']]).tolist()
    I_gids = np.concatenate([net.gid_ranges['L2_basket'], net.gid_ranges['L5_basket']]).tolist()

    EI_connections = pick_connection(net, src_gids=E_gids, target_gids=I_gids)
    EE_connections = pick_connection(net, src_gids=E_gids, target_gids=E_gids)
    II_connections = pick_connection(net, src_gids=I_gids, target_gids=I_gids)
    IE_connections = pick_connection(net, src_gids=I_gids, target_gids=E_gids)

    # Store in dictionary to be added to theta_dict
    theta_extra = {'EI_connections': EI_connections, 'EE_connections': EE_connections, 
                'II_connections': II_connections, 'IE_connections': IE_connections,
                'lamtha': 4.0}
    theta_extra['sample_idx'] =  sample_idx
    theta_dict['theta_extra'] = theta_extra

    beta_tuning_param_function(net, theta_dict, rate=rate)
    dpl = simulate_dipole(net, dt=0.5, tstop=1000, record_vsec='all', record_isec='all', record_dcell=True)

    g = net.cell_response.plot_spikes_raster(show=False)
    g.savefig(f'datasets_{suffix}/raster_plots/raster_{sample_idx}.png')
    plt.close()

    g = dpl[0].plot(show=False)
    g.savefig(f'datasets_{suffix}/dipole_plots/dipole_{sample_idx}.png')
    plt.close()

    g = dpl[0].plot_psd(fmin=0, fmax=100, show=False)
    g.savefig(f'datasets_{suffix}/psd_plots/psd_{sample_idx}.png')
    plt.close()

    np.save(f'datasets_{suffix}/dipole_data/dipole_{sample_idx}.npy', dpl[0].data['agg'], )

    for cell_type in net.cell_types.keys():
        if transform_dict is None:
            input_spike_scaler, vsec_scaler, isec_scaler = None, None, None
        else:
            input_spike_scaler = transform_dict[cell_type]['input_spike_scaler']
            vsec_scaler = transform_dict[cell_type]['vsec_scaler']
            isec_scaler = transform_dict[cell_type]['isec_scaler']

        training_set = utils.CellType_Dataset_Fast(
            net, cell_type=cell_type, window_size=500, data_step_size=250,
            input_spike_scaler=input_spike_scaler, vsec_scaler=vsec_scaler, isec_scaler=isec_scaler,
            soma_filter=True, device='cpu')
        torch.save(training_set, f'datasets_{suffix}/training_data/{cell_type}_dataset_{sample_idx}.pt')


# Generate subthreshold dataset
#------------------------------
net = calcium_model()

# Subthreshold
suffix = 'subthreshold'
rate = 20
prior_dict = {'EI_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'EE_gscale': {'bounds': (-2, 1), 'rescale_function': log_scale_forward},
              'II_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'IE_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'EI_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'EE_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'II_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'IE_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'L2e_distal': {'bounds': (-4, -3.5), 'rescale_function': log_scale_forward},
              'L2i_distal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'L5e_distal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'L5i_distal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'L2e_proximal': {'bounds': (-4, -3.5), 'rescale_function': log_scale_forward},
              'L2i_proximal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'L5e_proximal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'L5i_proximal': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              }


prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((n_sims,))

# First sample used to fit transformers
theta_samples[0,:] = torch.from_numpy(np.repeat(0.7, theta_samples.shape[1]))
run_hnn(theta_samples[0, :], 0, prior_dict, transform_dict=None, suffix=suffix)

transform_dict = {}
for cell_type in net.cell_types.keys():
    dataset = torch.load(f'datasets_subthreshold/training_data/{cell_type}_dataset_0.pt')
    transform_dict[cell_type] = {'input_spike_scaler': dataset.input_spike_scaler,
                                 'vsec_scaler': dataset.vsec_scaler,
                                 'isec_scaler': dataset.isec_scaler}
    
# Skip first sample which is used for creating transforms
Parallel(n_jobs=8)(delayed(run_hnn)(
    thetai, sample_idx+1, prior_dict, transform_dict, suffix) for
    (sample_idx, thetai) in enumerate(theta_samples[1:, :]))

# Generate suprathreshold dataset
#------------------------------
suffix = 'suprathreshold'
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((n_sims,))
lower_g, upper_g = -4, 0

update_keys = ['L2e_distal', 'L2i_distal', 'L5e_distal', 'L5i_distal',
               'L2e_proximal', 'L2i_proximal', 'L5e_proximal', 'L5i_proximal']
for key in update_keys:
    prior_dict[key]['bounds'] = (lower_g, upper_g)

Parallel(n_jobs=8)(delayed(run_hnn)(
    thetai, sample_idx, prior_dict, transform_dict, suffix) for
    (sample_idx, thetai) in enumerate(theta_samples))


# Generate connected dataset
#---------------------------
suffix = 'connected'
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((n_sims,))

Parallel(n_jobs=8)(delayed(run_hnn)(
    thetai, sample_idx, prior_dict, transform_dict, suffix) for
    (sample_idx, thetai) in enumerate(theta_samples))


