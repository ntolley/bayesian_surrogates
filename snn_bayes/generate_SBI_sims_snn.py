import sys
sys.path.append('../')
import os
import dill
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, beta_tuning_param_function, UniformPrior,
                   PriorBetaFiltered)
from hnn_core import calcium_model, pick_connection
from neurodsp.spectral import compute_spectrum, trim_spectrum
from torch import optim
from tqdm import tqdm
import joblib
import numpy as np

nsbi_sims = 10_000
tstop = 1000
num_prior_fits = 4
relative_beta_threshold = 0.3
dt = 0.05
fs = 1000 / dt

save_path = ''
save_suffix = 'sbi'

net = calcium_model()

cell_type_lookup = {
    'L2e': 'L2_pyramidal', 'L2i': 'L2_basket',
    'L5e': 'L5_pyramidal', 'L5i': 'L5_basket'}

cell_type_receptor = {
    'L2e': ['ampa', 'nmda'], 'L2i': ['gabaa', 'gabab'],
    'L5e': ['ampa', 'nmda'], 'L5i': ['gabaa', 'gabab']}

cell_types = ['L2e', 'L2i', 'L5e', 'L5i']
locations = ['proximal', 'distal']


theta_extra = {'cell_types': cell_types, 'cell_type_receptor': cell_type_receptor,
               'cell_type_lookup': cell_type_lookup, 'locations': locations,
               'invalid_conn_list': list(), 'valid_conn_list': list(),
               'lamtha': 4.0}
prior_dict = dict()

for src_type in cell_types:
    for target_type in cell_types:
        for receptor in cell_type_receptor[src_type]:
            for loc in locations:
                conn_name = f'{src_type}_{target_type}_{receptor}_{loc}'

                conn_indices = pick_connection(
                    net, src_gids=cell_type_lookup[src_type],
                    target_gids=cell_type_lookup[target_type],
                    receptor=receptor, loc=loc)

                if len(conn_indices) == 0:
                    theta_extra['invalid_conn_list'].append(conn_name)
                else:
                    assert len(conn_indices) == 1
                    theta_extra['valid_conn_list'].append(conn_name)
                    theta_extra[f'{conn_name}_conn_indices'] = conn_indices

                    prior_dict[f'{conn_name}_gbar'] = {'bounds': (-4, 0), 'rescale_function': log_scale_forward}
                    prior_dict[f'{conn_name}_prob'] = {'bounds': (0, 1), 'rescale_function': linear_scale_forward}

    for loc in locations:
        conn_name = f'{src_type}_{loc}'

        if conn_name != 'L5i_distal': # L5i_distal doesn't exist in the model
            prior_dict[f'{conn_name}_gbar'] = {'bounds': (-4, 0), 'rescale_function': log_scale_forward}

        

with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt, 'gid_ranges': net.gid_ranges,
                'num_prior_fits': num_prior_fits, 'relative_beta_threshold': relative_beta_threshold,
                'theta_extra': theta_extra}
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

start_cluster() # reserve resources for HNN simulations

for flow_idx in range(num_prior_fits):
    if flow_idx == 0:
        prior = UniformPrior(parameters=list(prior_dict.keys()))
    else:
        prior = prior_filtered
        
    theta_samples = prior.sample((nsbi_sims,))
    
    save_suffix = f'sbi_{flow_idx}'
    
    run_hnn_sim(net=net, param_function=beta_tuning_param_function, prior_dict=prior_dict,
                theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix,
                theta_extra=theta_extra)
    
    dpl_filter = np.load(f'{save_path}/sbi_sims/dpl_sbi_{flow_idx}.npy')
    theta_filter = np.load(f'{save_path}/sbi_sims/theta_sbi_{flow_idx}.npy')
    
    times = np.linspace(0, tstop / 1000, dpl_filter.shape[1])
    times_mask = times > 0.2 # burn in time
    
    freqs, powers = compute_spectrum(dpl_filter[:, times_mask], fs, method='welch', avg_type='median')
    freqs, powers = trim_spectrum(freqs, powers, [0, 100])
    
    beta_mask = (freqs > 13) & (freqs < 30)
    beta_power = powers[:, beta_mask].sum(axis=1)
    relative_beta = beta_power / powers.sum(axis=1)
    
    relative_beta_mask = relative_beta > relative_beta_threshold
    
    dpl_filter = dpl_filter[relative_beta_mask, :]
    theta_filter = theta_filter[relative_beta_mask, :]

    prior_filtered = PriorBetaFiltered(parameters=list(prior_dict.keys()))
    optimizer = optim.Adam(prior_filtered.flow.parameters())

    num_iter = 5000
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()
        loss = -prior_filtered.flow.log_prob(inputs=theta_filter).mean()
        loss.backward()
        optimizer.step()
    state_dict = prior_filtered.flow.state_dict()
    joblib.dump(state_dict, f'{save_path}/flows/prior_filtered_flow_{flow_idx}.pkl')


os.system('scancel -u ntolley')