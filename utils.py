import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
import multiprocessing
from joblib import Parallel, delayed
import pickle
from hnn_core.cell import _get_gaussian_connection
from hnn_core.network import _connection_probability
from sbi import utils as sbi_utils
from typing import Dict, Tuple, Optional, List




#Helper function to pytorch train networks for decoding
# def train_model(model, optimizer, criterion, max_epochs, training_generator, device, print_freq=10):
#     train_loss_array = []
#     model.train()
#     # Loop over epochs
#     for epoch in range(max_epochs):
#         train_batch_loss = []
#         for batch_x, batch_y in training_generator:
#             optimizer.zero_grad() # Clears existing gradients from previous epoch
#             batch_x = batch_x.float().to(device)
#             batch_y = batch_y.float().to(device)
#             output = model(batch_x)
#             train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
#             train_loss.backward() # Does backpropagation and calculates gradients
#             optimizer.step() # Updates the weights accordingly

#             train_batch_loss.append(train_loss.item())
#         print('*',end='')
#         train_loss_array.append(train_batch_loss)
#         #Print Loss
#         if (epoch+1)%print_freq == 0:
#             print('')
#             print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
#             print('Train Loss: ' + str(np.mean(train_batch_loss)))
#     return train_loss_array

#Helper function to pytorch train networks for decoding
def train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, print_freq=10, early_stop=20):
    train_loss_array = []
    validation_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        for batch_x, batch_y in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
            # train_loss = criterion(output[:,:,:], batch_y[:,:,:])

            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate train set predictions
            for batch_x, batch_y in validation_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
                # validation_loss = criterion(output[:,:,:], batch_y[:,:,:])

                validation_batch_loss.append(validation_loss.item())

        validation_loss_array.append(validation_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            
        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.4f}  ... Validation Loss: {:.4f}'.format(train_epoch_loss,validation_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'max_epochs':max_epochs}
    return loss_dict

#Helper function to pytorch train networks for decoding
# def train_validate_test_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, testing_generator,device, print_freq=10, early_stop=20):
#     train_loss_array = []
#     validation_loss_array = []
#     test_loss_array = []
#     # Loop over epochs
#     min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
#     for epoch in range(max_epochs):
#         #___Train model___
#         model.train()
#         train_batch_loss = []
#         validation_batch_loss = []
#         test_batch_loss = []
#         for batch_x, batch_y in training_generator:
#             optimizer.zero_grad() # Clears existing gradients from previous epoch
#             batch_x = batch_x.float().to(device)
#             batch_y = batch_y.float().to(device)
#             output = model(batch_x)
#             train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
#             # train_loss = criterion(output[:,:,:], batch_y[:,:,:])
#             train_loss.backward() # Does backpropagation and calculates gradients
#             optimizer.step() # Updates the weights accordingly

#             train_batch_loss.append(train_loss.item())
        
#         train_loss_array.append(train_batch_loss)

#         #___Evaluate Model___
#         with torch.no_grad():
#             model.eval()
#             #Generate validation set predictions
#             for batch_x, batch_y in validation_generator:
#                 batch_x = batch_x.float().to(device)
#                 batch_y = batch_y.float().to(device)
#                 output = model(batch_x)
#                 validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

#                 validation_batch_loss.append(validation_loss.item())

#             validation_loss_array.append(validation_batch_loss)

#             #Generate test set predictions
#             for batch_x, batch_y in testing_generator:
#                 batch_x = batch_x.float().to(device)
#                 batch_y = batch_y.float().to(device)
#                 output = model(batch_x)
#                 test_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

#                 test_batch_loss.append(test_loss.item())

#             test_loss_array.append(test_batch_loss)

#         #Compute average loss on batch
#         train_epoch_loss = np.mean(train_batch_loss)
#         train_epoch_std = np.std(train_batch_loss)
#         validation_epoch_loss = np.mean(validation_batch_loss)
#         validation_epoch_std = np.std(validation_batch_loss)
#         test_epoch_loss = np.mean(test_batch_loss)
#         test_epoch_std = np.std(test_batch_loss)

#        #Check if validation loss reaches minimum 
#         if validation_epoch_loss < min_validation_loss:
#             print('*',end='')
#             min_validation_loss = np.copy(validation_epoch_loss)
#             min_validation_std = np.copy(validation_epoch_std)
#             min_validation_counter = 0
#             min_validation_epoch = np.copy(epoch+1)

#             min_train_loss = np.copy(train_epoch_loss)
#             min_train_std = np.copy(train_epoch_std)
#             min_test_loss = np.copy(test_epoch_loss)
#             min_test_std = np.copy(test_epoch_std)


#         else:
#             print('.',end='')
#             min_validation_counter += 1

#         #Print Loss Scores
#         if (epoch+1)%print_freq == 0:
#             print('')
#             print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
#             print('Train Loss: {:.2f}  ... Validation Loss: {:.2f} ... Test Loss: {:.2f}'.format(train_epoch_loss, validation_epoch_loss, test_epoch_loss))
        
#         #Early stop if no validation improvement over set number of epochs
#         if min_validation_counter > early_stop:
#             print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
#             break

#     loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
#     'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
#     'min_test_loss':min_test_loss, 'min_test_std':min_test_std,
#     'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'test_loss_array':test_loss_array, 'max_epochs':max_epochs}
#     return loss_dict

#Vectorized correlation coefficient of two matrices on specified dimension
def matrix_corr(x, y, axis=0):
    num_tpts, _ = np.shape(x)
    mean_x, mean_y = np.tile(np.mean(x, axis=axis), [num_tpts,1]), np.tile(np.mean(y, axis=axis), [num_tpts,1])
    corr = np.sum(np.multiply((x-mean_x), (y-mean_y)), axis=axis) / np.sqrt(np.multiply( np.sum((x-mean_x)**2, axis=axis), np.sum((y-mean_y)**2, axis=axis) ))
    return corr

#Helper function to evaluate decoding performance on a trained model
def evaluate_model(model, generator, device):
    #Run model through test set
    with torch.no_grad():
        model.eval()
        #Generate predictions
        y_pred_tensor = torch.zeros(len(generator.dataset),  generator.dataset[0][1].shape[1])
        batch_idx = 0
        for batch_x, batch_y in generator:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            y_pred_tensor[batch_idx:(batch_idx+output.size(0)),:] = output[:,-1,:]
            batch_idx += output.size(0)

    y_pred = y_pred_tensor.detach().cpu().numpy()
    return y_pred


class CellType_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, net, cell_type='L5_pyramidal', data_step_size=1,
                 window_size=100, input_spike_scaler=None, vsec_scaler=None, isec_scaler=None,
                 soma_filter=False, device='cpu'):
        
        network_data = Network_Data(net, soma_filter=soma_filter)
        self.cell_type = cell_type
        self.num_cells = len(network_data.net.gid_ranges[self.cell_type])
        self.data_step_size = data_step_size
        self.window_size = window_size + 1  # Used to offset the target by 1 timestep
        self.device = device

        self.vsec_names = network_data.neuron_data_dict[list(net.gid_ranges[cell_type])[0]].vsec_names
        self.isec_names = network_data.neuron_data_dict[list(net.gid_ranges[cell_type])[0]].isec_names

        self.input_spike_list, self.vsec_list, self.isec_list = self.process_data(network_data)
        assert len(self.input_spike_list) == len(self.vsec_list) == len(self.isec_list) == self.num_cells

        if input_spike_scaler is None:
            self.input_spike_scaler = StandardScaler()
            self.input_spike_scaler.fit(np.vstack(self.input_spike_list))
        else:
            self.input_spike_scaler = input_spike_scaler
        
        if vsec_scaler is None:
            self.vsec_scaler = StandardScaler()
            self.vsec_scaler.fit(np.vstack(self.vsec_list))
        else:
            self.vsec_scaler = vsec_scaler
        
        if isec_scaler is None:
            self.isec_scaler = StandardScaler()
            self.isec_scaler.fit(np.vstack(self.isec_list))
        else:
            self.isec_scaler = isec_scaler
 

        self.input_spike_list, self.vsec_list, self.isec_list, self.slice_lookup = self.transform_data()
        self.num_samples = len(self.input_spike_list) * (self.input_spike_list[0].shape[0] - self.window_size)

        assert self.num_samples == len(self.slice_lookup)

        self.X_data = torch.concat(self.input_spike_list, dim=0)
        self.y_data = torch.concat(self.vsec_list, dim=0)
    
    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        data_index = self.slice_lookup[slice_index].astype(int)
        if np.isscalar(data_index):
            data_index = np.array([data_index])

        
        X, y = list(), list()
        for idx in data_index:
            X.append(self.X_data[idx - self.window_size: idx, :])
            y.append(self.y_data[idx - self.window_size: idx, :])

        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        
        return X[:, :-1, :].squeeze(dim=0), y[:, 1:, :].squeeze(dim=0)
    

    def process_data(self, network_data):
        gid_list = network_data.net.gid_ranges[self.cell_type]
        input_spike_list, vsec_list, isec_list = list(), list(), list()
        for gid in gid_list:
            input_spike_list.append(network_data.input_spike_dict[gid].T)
            vsec_list.append(network_data.neuron_data_dict[gid].vsec_array.T)
            isec_list.append(network_data.neuron_data_dict[gid].isec_array.T)
        
        return input_spike_list, vsec_list, isec_list

    def transform_data(self):
        sim_len = self.input_spike_list[0].shape[0]
        input_spike_transform_list, vsec_transform_list, isec_transform_list = list(), list(), list()

        slice_lookup = list()
        for idx in range(self.num_cells):
            input_spike_transformed = self.input_spike_scaler.transform(self.input_spike_list[idx])
            input_spike_transformed = torch.from_numpy(input_spike_transformed)
            input_spike_transform_list.append(input_spike_transformed)


            vsec_transformed = self.vsec_scaler.transform(self.vsec_list[idx])
            vsec_transformed = torch.from_numpy(vsec_transformed)
            vsec_transform_list.append(vsec_transformed)

            isec_transformed = self.isec_scaler.transform(self.isec_list[idx])
            isec_transformed = torch.from_numpy(isec_transformed)
            isec_transform_list.append(isec_transformed)

            sim_len = input_spike_transformed.shape[0]
            assert sim_len == isec_transformed.shape[0] == vsec_transformed.shape[0] == isec_transformed.shape[0]

            slice_lookup_temp = np.arange(self.window_size, sim_len, self.data_step_size) + (idx * sim_len)
            slice_lookup.extend(slice_lookup_temp)

        slice_lookup = np.array(slice_lookup)

        return input_spike_transform_list, vsec_transform_list, isec_transform_list, slice_lookup


        

class SingleNeuron_Data:
    #'Characterizes a dataset for PyTorch'
    def __init__(self, net, gid, soma_filter=False):
        self.gid = gid
        self.cell_type = net.gid_to_type(self.gid)

        if self.cell_type in net.cell_types:
            self.is_cell = True

            # Get voltages
            if soma_filter == True:
                vsec_list = [net.cell_response.vsec[0][gid]['soma']]
                vsec_names = ['soma']
            elif soma_filter == False:
                vsec_list, vsec_names = list(), list()
                for sec_name, vsec in net.cell_response.vsec[0][gid].items():
                    vsec_list.append(vsec)
                    vsec_names.append(sec_name)

            # Add dipole to end of voltage list
            if self.cell_type in ['L5_pyramidal', 'L2_pyramidal']:
                dcell = net.cell_response.dcell[0][gid]
                vsec_names.append('dipole')
                vsec_list.append(dcell)

            self.vsec_names = vsec_names
            self.vsec_array = np.array(vsec_list)

            isec_list, isec_names = list(), list()
            # Get currents
            for sec_name, isec_dict in net.cell_response.isec[0][gid].items():
                for isec_name, isec in isec_dict.items():
                    isec_list.append(isec)
                    isec_names.append(isec_name)

            self.isec_names = isec_names
            self.isec_array = np.array(isec_list)

            # Create dictionary to look up row for each section/receptor combo
            isec_name_lookup = {name: idx for idx, name in enumerate(isec_names)}
            self.isec_name_lookup = isec_name_lookup
            
        else:
            self.is_cell = False
            
            
        self.spikes_binned = self.get_binned_spikes(net)

    def get_binned_spikes(self, net):
        spike_times = np.array(net.cell_response.spike_times)
        spike_gids = np.array(net.cell_response.spike_gids)

        spike_times_gid = spike_times[spike_gids == self.gid]
        spikes_binned = np.histogram(spike_times_gid, bins=net.cell_response.times)[0]

        spikes_binned = np.concatenate([spikes_binned, [0.0]])
        return spikes_binned
        

class Network_Data:
    def __init__(self, net, soma_filter=False):
        self.net = net
        self.dt = net.cell_response.times[1] - net.cell_response.times[0]
        self.neuron_data_dict = dict()
        self.input_spike_dict = dict()
        
        self.connectivity_dict = dict()
        for cell_type, gid_list in net.gid_ranges.items():

            for gid in gid_list:
                self.neuron_data_dict[gid] = SingleNeuron_Data(net, gid, soma_filter=soma_filter)

                # Initialize blank arrays for spikes recieved by each cell
                if cell_type in net.cell_types:
                    self.input_spike_dict[gid] = np.zeros(self.neuron_data_dict[gid].isec_array.shape)

            # Initialize blank arrays for connectivity
            if cell_type in net.cell_types:
                input_size = self.neuron_data_dict[gid].isec_array.shape[0]  # Just need first cell for num inputs
                # Dictionary indexed by target type, entry is (num_targets, num_inputs, num_sources) shape matrix
                self.connectivity_dict[cell_type] = np.zeros((len(net.gid_ranges[cell_type]), input_size, net._n_gids))

        self.delay_matrix = np.zeros((net._n_gids, net._n_cells))
        for conn in net.connectivity:
            for src_gid, target_gid_list in conn['gid_pairs'].items():
                src_type = conn['src_type']

                # Positions assigned based on gid in real network
                src_pos_list = list(net.pos_dict[src_type])

                # Proximal/distal drives
                if src_type in net.cell_types:
                    src_pos = src_pos_list[src_gid - list(net.gid_ranges[src_type])[0]]
                else:
                    src_pos = src_pos_list[0]

                # Loop through all target gids and append spikes to appropriate array
                for target_gid in target_gid_list:
                    conn_spikes = self.neuron_data_dict[src_gid].spikes_binned

                    target_type = conn['target_type']
                    receptor = conn['receptor']
                    loc = conn['loc']
                    # Positions assigned based on gid in real network
                    target_pos_list = list(net.pos_dict[target_type])
                    if target_type in net.cell_types:
                        target_pos = target_pos_list[target_gid - list(net.gid_ranges[target_type])[0]]
                    else:
                        target_pos = target_pos_list[0]

                    # Get distance dependent weight/delay for connection
                    weight, delay = _get_gaussian_connection(
                        src_pos, target_pos, conn['nc_dict'], inplane_distance=net._inplane_distance)

                    delay_samples = int(round(delay / self.dt))
                    # if delay_samples == 0:
                    #     delay_samples = 1 # Can only deliver a spike at least 1 time step in the future

                    # Add delay to delay matrix
                    self.delay_matrix[src_gid, target_gid] = delay                    

                    # Delay spikes by delay_samples
                    # conn_spikes = np.concatenate([np.zeros((delay_samples,)), conn_spikes])[delay_samples:]
                    conn_spikes *= weight

                    if loc in net.cell_types[target_type].sect_loc:
                        sect_loc = net.cell_types[target_type].sect_loc[loc]
                    else:
                        sect_loc = [loc]
                        assert loc in net.cell_types[target_type].sections

                    for sec in sect_loc:
                        input_spike_name = f'{sec}_{receptor}'
                        input_sec_idx = self.neuron_data_dict[target_gid].isec_name_lookup[input_spike_name]
                        if delay_samples != 0:
                            self.input_spike_dict[target_gid][input_sec_idx, delay_samples:] += conn_spikes[:-delay_samples]
                        else:
                            self.input_spike_dict[target_gid][input_sec_idx, :] += conn_spikes

                        self.connectivity_dict[target_type][target_gid - target_gid_list[0], input_sec_idx, src_gid] = weight

            

class CellType_Dataset_Fast(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, net, cell_type='L5_pyramidal', data_step_size=1,
                 window_size=100, input_spike_scaler=None, vsec_scaler=None, isec_scaler=None,
                 soma_filter=False, device='cpu'):
        
        network_data = Network_Data(net, soma_filter=soma_filter)
        self.connectivity_dict = network_data.connectivity_dict
        self.delay_matrix = network_data.delay_matrix

        self.cell_type = cell_type
        self.num_cells = len(network_data.net.gid_ranges[self.cell_type])
        self.data_step_size = data_step_size
        self.window_size = window_size
        self.device = device

        self.vsec_names = network_data.neuron_data_dict[list(net.gid_ranges[cell_type])[0]].vsec_names
        self.isec_names = network_data.neuron_data_dict[list(net.gid_ranges[cell_type])[0]].isec_names

        self.input_spike_list, self.vsec_list, self.isec_list = self.process_data(network_data)
        assert len(self.input_spike_list) == len(self.vsec_list) == len(self.isec_list) == self.num_cells

        if input_spike_scaler is None:
            self.input_spike_scaler = MinMaxScaler(feature_range=(0, 1))
            # self.input_spike_scaler = StandardScaler()
            self.input_spike_scaler.fit(np.vstack(self.input_spike_list))
        else:
            self.input_spike_scaler = input_spike_scaler
        
        if vsec_scaler is None:
            self.vsec_scaler = StandardScaler()
            self.vsec_scaler.fit(np.vstack(self.vsec_list))
        else:
            self.vsec_scaler = vsec_scaler
        
        if isec_scaler is None:
            self.isec_scaler = StandardScaler()
            self.isec_scaler.fit(np.vstack(self.isec_list))
        else:
            self.isec_scaler = isec_scaler
 

        self.input_spike_unfolded, self.vsec_unfolded, self.isec_unfolded = self.unfold_data()
        
        # X is one step behind y
        self.X_tensor = self.input_spike_unfolded[:, :-1, :]
        self.y_tensor = self.vsec_unfolded[:, 1:, :]
        assert self.X_tensor.shape[0] == self.y_tensor.shape[0]
        self.num_samples = self.X_tensor.shape[0]

        self.X_tensor = self.X_tensor.float().to(self.device)
        self.y_tensor = self.y_tensor.float().to(self.device)

    
    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        return self.X_tensor[slice_index,:,:], self.y_tensor[slice_index,:,:]
    

    def process_data(self, network_data):
        gid_list = network_data.net.gid_ranges[self.cell_type]
        input_spike_list, vsec_list, isec_list = list(), list(), list()
        for gid in gid_list:
            input_spike_list.append(network_data.input_spike_dict[gid].T)
            vsec_list.append(network_data.neuron_data_dict[gid].vsec_array.T)
            isec_list.append(network_data.neuron_data_dict[gid].isec_array.T)
        
        return input_spike_list, vsec_list, isec_list

    def unfold_data(self):
        input_spike_unfold_list, vsec_unfold_list, isec_unfold_list = list(), list(), list()
        for idx in range(self.num_cells):
            # Input spikes
            input_spike_transformed = self.input_spike_scaler.transform(self.input_spike_list[idx])
            input_spike_transformed = torch.from_numpy(input_spike_transformed)
            input_spike_unfolded = input_spike_transformed.unfold(0, self.window_size + 1, self.data_step_size).transpose(1,2)
            input_spike_unfold_list.append(input_spike_unfolded)

            # Voltages
            vsec_transformed = self.vsec_scaler.transform(self.vsec_list[idx])
            vsec_transformed = torch.from_numpy(vsec_transformed)
            vsec_unfolded = vsec_transformed.unfold(0, self.window_size + 1, self.data_step_size).transpose(1,2)
            vsec_unfold_list.append(vsec_unfolded)

            # Currents
            isec_transformed = self.isec_scaler.transform(self.isec_list[idx])
            isec_transformed = torch.from_numpy(isec_transformed)
            isec_unfolded = isec_transformed.unfold(0, self.window_size + 1, self.data_step_size).transpose(1,2)
            isec_unfold_list.append(isec_unfolded)

        input_spike_unfolded = torch.concat(input_spike_unfold_list, dim=0)
        vsec_unfolded = torch.concat(vsec_unfold_list, dim=0)
        isec_unfolded = torch.concat(isec_unfold_list, dim=0)

        return input_spike_unfolded, vsec_unfolded, isec_unfolded

def linear_scale_forward(value, bounds, constrain_value=True):
    """Scale value in range (0,1) to range bounds"""
    if constrain_value:
        assert np.all(value >= 0.0) and np.all(value <= 1.0)
        
    assert isinstance(bounds, tuple)
    assert bounds[0] < bounds[1]
    
    return (bounds[0] + (value * (bounds[1] - bounds[0]))).astype(float)

def linear_scale_array(value, bounds, constrain_value=True):
    """Scale columns of array according to bounds"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [linear_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def log_scale_forward(value, bounds, constrain_value=True):
    """log scale value in range (0,1) to range bounds in base 10"""
    rescaled_value = linear_scale_forward(value, bounds, constrain_value)
    
    return 10**rescaled_value

def log_scale_array(value, bounds, constrain_value=True):
    """log scale columns of array according to bounds in base 10"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [log_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

class UniformPrior(sbi_utils.BoxUniform):
    """Prior distribution object that generates uniform sample on range (0,1)"""
    def __init__(self, parameters):
        """
        Parameters
        ----------
        parameters: list of str
            List of parameter names for prior distribution
        """
        self.parameters = parameters
        low = len(parameters)*[0]
        high = len(parameters)*[1]
        super().__init__(low=torch.tensor(low, dtype=torch.float32),
                         high=torch.tensor(high, dtype=torch.float32))

# Poisson drive to all synapses and random connections
def section_drive_param_function(net, theta_dict, rate=10):
    seed_rng = np.random.default_rng(theta_dict['theta_extra']['sample_idx'])
    seed_array = seed_rng.integers(10e5, size=100)

    seed_count = 0
    
    # Update prob and gbar of network connections
    valid_conn_list = theta_dict['theta_extra']['valid_conn_list']
    for conn_name in valid_conn_list:
    
        conn_prob_name = f'{conn_name}_prob'
        conn_gbar_name = f'{conn_name}_gbar'
        
        conn_idx = theta_dict['theta_extra'][f'{conn_name}_conn_idx']
        
        probability = theta_dict[conn_prob_name]
        gbar = theta_dict[conn_gbar_name]
        
        # Prune connections using internal connection_probability function
        _connection_probability(
            net.connectivity[conn_idx], probability=probability, conn_seed=seed_array[seed_count])
        net.connectivity[conn_idx]['probability'] = probability
        net.connectivity[conn_idx]['nc_dict']['A_weight'] = gbar
        seed_count = seed_count + 1

    for conn_idx in range(len(net.connectivity)):
        # net.connectivity[conn_idx]['nc_dict']['lamtha'] = theta_dict['theta_extra']['lamtha']
        net.connectivity[conn_idx]['nc_dict']['lamtha'] = theta_dict['lamtha']

    # Add drives
    valid_drive_dict = theta_dict['theta_extra']['valid_drive_dict']
    for drive_name in valid_drive_dict.keys():
        cell_type = valid_drive_dict[drive_name]['cell_type']
        location = valid_drive_dict[drive_name]['location']
        receptor = valid_drive_dict[drive_name]['receptor']

        drive_prob_name = f'{drive_name}_prob'
        drive_gbar_name = f'{drive_name}_gbar'

        probability = theta_dict[drive_prob_name]
        gbar = theta_dict[drive_gbar_name]

        weights_ampa, weights_nmda, weights_gabaa, weights_gabab = None, None, None, None
        if receptor == 'ampa':
            weights_ampa = {cell_type: gbar}
        elif receptor == 'nmda':
            weights_nmda = {cell_type: gbar}
        elif receptor == 'gabaa':
            weights_gabaa = {cell_type: gbar}
        elif receptor == 'gabab':
            weights_gabab = {cell_type: gbar}

        net.add_poisson_drive(
            name=drive_name, tstart=0, tstop=None, rate_constant=rate, location=location, n_drive_cells='n_cells',
            cell_specific=True, weights_ampa=weights_ampa, weights_nmda=weights_nmda,
            weights_gabaa=weights_gabaa, weights_gabab=weights_gabab, space_constant=1e50,
            synaptic_delays=0.0, probability=probability, event_seed=seed_array[-1], conn_seed=seed_array[-2])


# def beta_tuning_param_function(net, theta_dict, rate=10):
#     conn_type_list = {'EI_connections': 'EI', 'EE_connections': 'EE', 
#                       'II_connections': 'II', 'IE_connections': 'IE'}
    
#     seed_rng = np.random.default_rng(theta_dict['theta_extra']['sample_idx'])
#     seed_array = seed_rng.integers(10e5, size=100)

#     seed_count = 0
#     for conn_type_name, conn_suffix in conn_type_list.items():
#         conn_prob_name = f'{conn_suffix}_prob'
#         conn_gscale_name = f'{conn_suffix}_gscale'
        
#         conn_indices = theta_dict['theta_extra'][conn_type_name]
#         probability = theta_dict[conn_prob_name]
#         gscale = theta_dict[conn_gscale_name]
        
#         for conn_idx in conn_indices:
#             # Prune connections using internal connection_probability function
#             _connection_probability(
#                 net.connectivity[conn_idx], probability=probability, conn_seed=seed_array[seed_count])
#             net.connectivity[conn_idx]['probability'] = probability
#             net.connectivity[conn_idx]['nc_dict']['A_weight'] *= gscale
#             seed_count = seed_count + 1

#     for conn_idx in range(len(net.connectivity)):
#         net.connectivity[conn_idx]['nc_dict']['lamtha'] = theta_dict['theta_extra']['lamtha']          
        
#     # rate = 10
#     # Add Poisson drives
#     weights_ampa_d1 = {'L2_pyramidal': theta_dict['L2e_distal'], 'L5_pyramidal': theta_dict['L5e_distal'],
#                        'L2_basket': theta_dict['L2i_distal']}
#     rates_d1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate}

#     net.add_poisson_drive(
#         name='distal', tstart=0, tstop=None, rate_constant=rates_d1, location='distal', n_drive_cells='n_cells',
#         cell_specific=True, weights_ampa=weights_ampa_d1, weights_nmda=None, space_constant=1e50,
#         synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-1], conn_seed=seed_array[-2])

#     weights_ampa_p1 = {'L2_pyramidal': theta_dict['L2e_proximal'], 'L5_pyramidal': theta_dict['L5e_proximal'],
#                        'L2_basket': theta_dict['L2i_proximal'], 'L5_basket': theta_dict['L5i_proximal']}
#     rates_p1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate, 'L5_basket': rate}

#     net.add_poisson_drive(
#         name='proximal', tstart=0, tstop=None, rate_constant=rates_p1, location='proximal', n_drive_cells='n_cells',
#         cell_specific=True, weights_ampa=weights_ampa_p1, weights_nmda=None, space_constant=1e50,
#         synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-3], conn_seed=seed_array[-4])


#LSTM/GRU architecture for decoding
class model_celltype_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64, n_layers=5, dropout=0.1, kernel_size=200, device='cuda:0', bidirectional=False):
        super(model_celltype_lstm, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers * num_directions
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.kernel_size = kernel_size

        self.tau1_init, self.tau2_init = 10, 20

        self.kernel_scale_init, self.kernel_offset_init = 10, -5
        self.kernel = self.get_kernel(torch.arange(0, self.kernel_size, 1).to(self.device),
                                tau1=self.tau1_init, tau2=self.tau2_init).float().flip(0)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.lstm.flatten_parameters()

        self.fc_input = nn.Sequential(
            nn.Linear(input_size, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )

        self.fc_output = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim*num_directions, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)

        )
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        kernel_product = self.kernel.tile(dims=(batch_size, self.input_size, 1)).transpose(1,2)

        out = (kernel_product * x).sum(dim=1).unsqueeze(1)
        
        # out = self.fc_input(out.contiguous())
        out, (h0, c0) = self.lstm(out, (h0, c0))
        out = out.contiguous()
        out = self.fc_output(out)
            
        return (out, h0, c0)
        # return out_hidden

    def get_kernel(self, t_vec, tau1=10, tau2=20):
        G = tau2/(tau2-tau1)*(-torch.exp(-t_vec/tau1) + torch.exp(-t_vec/tau2))
        return G



#LSTM/GRU architecture for decoding
class model_network_custom_weights(nn.Module):
    def __init__(self, net, L5Pyr_model, training_set, n_L5Pyr=100, n_inhib=30, soma_idx=0, device='cuda:0', bidirectional=False):
        super(model_network_custom_weights, self).__init__()
        self.net = net
        self.L5Pyr_model = L5Pyr_model
        self.training_set = training_set.datasets[0]
        self.device = device

        self.n_L5Pyr = n_L5Pyr
        self.n_inhib = n_inhib
        # self.n_inhib = 0

        self.L5Pyr_gids = torch.tensor(list(range(0,n_L5Pyr)))
        self.inhib_gids = torch.tensor(list(range(n_L5Pyr, n_L5Pyr+n_inhib)))
        self.n_cells = n_L5Pyr + n_inhib

        self.drive_target_gids = None


        self.threshold = 100
        self.soma_idx = soma_idx  # index of soma compartment

        self.prox_indices = self.get_drive_indices('proximal', 'ampa', self.training_set, self.net)
        self.inhib_indices = self.get_drive_indices('soma', 'gabaa', self.training_set, self.net)

        self.conn_mean = torch.tensor([10.0, 10.0, 10.0, 10.0])
        self.conn_std = torch.tensor([1.0, 1.0, 1.0, 1.0])

        weight_matrix =  self.init_weight_matrix().detach().requires_grad_(True)
        # weight_matrix = torch.zeros((self.n_cells, self.L5Pyr_model.input_size, self.n_cells)).float().to(self.device)
    
        self.weight_matrix = nn.Parameter(weight_matrix).detach().to(self.device).requires_grad_(True)

    def forward(self, input_spikes_tensor):
        pred_y = list()
        hidden = self.L5Pyr_model.init_hidden(input_spikes_tensor.size(0))
        
        for time_idx in range(self.L5Pyr_model.kernel_size, input_spikes_tensor.size(1)-1):
            batch_x = input_spikes_tensor[:, time_idx-self.L5Pyr_model.kernel_size:time_idx, :].to(self.device)

            out, hidden = self.L5Pyr_model(batch_x, hidden)
            pred_y.append(out[:,-1, self.soma_idx])

            if time_idx > self.L5Pyr_model.kernel_size:
                spike_mask = ((pred_y[-1][:] > self.threshold) & (pred_y[-2][:] < self.threshold))
            
                input_spikes_tensor[:, time_idx+1, :] += torch.matmul(self.weight_matrix.clamp(0, 10), spike_mask.float())

        pred_y = torch.stack(pred_y)
        return pred_y


    def init_weight_matrix(self):
        weight_matrix = torch.zeros((self.n_cells, self.L5Pyr_model.input_size, self.n_cells)).float().to(self.device)

        # EE
        weight_matrix = self.random_weight_matrix(n_cells=self.n_cells, n_sec=self.L5Pyr_model.input_size, sec_indices=self.prox_indices,
                                     src_size=self.n_L5Pyr, target_size=30, src_range=self.L5Pyr_gids, target_range=self.L5Pyr_gids,
                                     weight_matrix=weight_matrix, mean_weight=self.conn_mean[0])
       # EI
        weight_matrix = self.random_weight_matrix(n_cells=self.n_cells, n_sec=self.L5Pyr_model.input_size, sec_indices=self.prox_indices,
                                                src_size=self.n_L5Pyr, target_size=30, src_range=self.L5Pyr_gids, target_range=self.inhib_gids,
                                                weight_matrix=weight_matrix, mean_weight=self.conn_mean[1])

        # IE
        weight_matrix = self.random_weight_matrix(n_cells=self.n_cells, n_sec=self.L5Pyr_model.input_size, sec_indices=self.inhib_indices,
                                                src_size=self.n_inhib, target_size=100, src_range=self.inhib_gids, target_range=self.L5Pyr_gids,
                                                weight_matrix=weight_matrix, mean_weight=self.conn_mean[2])

        # II
        weight_matrix = self.random_weight_matrix(n_cells=self.n_cells, n_sec=self.L5Pyr_model.input_size, sec_indices=self.inhib_indices,
                                                src_size=self.n_inhib, target_size=30, src_range=self.inhib_gids, target_range=self.inhib_gids,
                                                weight_matrix=weight_matrix, mean_weight=self.conn_mean[3])


        return weight_matrix

    def random_weight_matrix(self, n_cells, n_sec, sec_indices, src_size=100, target_size=100,
                           src_range=None, target_range=None, mean_weight=5., weight_matrix=None):
        if weight_matrix is None:
            weight_matrix = torch.zeros((n_cells, n_sec, n_cells)).float()
        
        if src_range is None:
            src_range = list(range(n_cells))

        if target_range is None:
            target_range = list(range(n_cells))

        # src_gid_list = np.random.choice(src_range, size=src_size, replace=False)
        src_probs = torch.ones(size=(len(src_range),))
        src_indices = torch.multinomial(src_probs, num_samples=src_size, replacement=False)
        src_gid_list = torch.index_select(src_range, dim=0, index=src_indices)

        for src_gid in src_gid_list:
            target_probs = torch.ones(size=(len(target_range),))
            target_indices = torch.multinomial(target_probs, num_samples=target_size, replacement=False)
            target_gid_list = torch.index_select(target_range, dim=0, index=target_indices)

            for target_gid in target_gid_list:
                weight_matrix[target_gid, sec_indices, src_gid] += torch.normal(mean=mean_weight, std=1.0)
        return weight_matrix

    # Create input spike train
    def init_input_spikes(self, n_samples=1000):
        input_spikes = torch.zeros((self.n_cells, n_samples + 1, len(self.training_set.isec_names)))

        # prox_time_indices = np.arange(0, n_samples, 1000)  # 1 Hz input

        self.drive_target_gids = np.random.choice(self.L5Pyr_gids, size=30, replace=False)
        for gid in self.drive_target_gids:
            prox_time_indices = np.random.choice(list(range(0, n_samples)), size=100, replace=False)
            # prox_time_indices = torch.randint(0, n_samples, size=(100,), requires_grad=True, )
            drive_weight = np.random.uniform(0, 10)
            for isec_idx in self.prox_indices:
                input_spikes[gid, prox_time_indices, isec_idx] = drive_weight

        return input_spikes

    def get_drive_indices(drive_name, receptor, training_set, net):
        if drive_name == 'soma':
            drive_loc = ['soma']
        else:
            drive_loc = net.cell_types['L5_pyramidal'].sect_loc[drive_name]
        syn_names = [f'{loc}_{receptor}' for loc in drive_loc]
        syn_indices = np.where(np.in1d(training_set.isec_names, syn_names))[0]

        return syn_indices

class make_connectivity_matrix:
    def __init__(self, net, soma_filter=False):
        self.net = net
        self.dt = 0.5
        self.input_spike_dict = dict()
        self.neuron_info_dict = self.make_neuron_info()
        
        self.connectivity_dict = dict()
        for cell_type, gid_list in net.gid_ranges.items():

            # Initialize blank arrays for connectivity
            if cell_type in net.cell_types:
                input_size = len(self.neuron_info_dict[cell_type]['isec'])  # Just need first cell for num inputs
                # Dictionary indexed by target type, entry is (num_targets, num_inputs, num_sources) shape matrix
                self.connectivity_dict[cell_type] = np.zeros((len(net.gid_ranges[cell_type]), input_size, net._n_gids))

                isec_names = self.neuron_info_dict[cell_type]['isec']
                self.neuron_info_dict[cell_type]['isec_name_lookup'] =  {name: idx for idx, name in enumerate(isec_names)}

        self.delay_matrix = np.zeros((net._n_gids, net._n_cells))
        for conn in net.connectivity:
            for src_gid, target_gid_list in conn['gid_pairs'].items():
                src_type = conn['src_type']

                # Positions assigned based on gid in real network
                src_pos_list = list(net.pos_dict[src_type])

                # Proximal/distal drives
                if src_type in net.cell_types:
                    src_pos = src_pos_list[src_gid - list(net.gid_ranges[src_type])[0]]
                else:
                    src_pos = src_pos_list[0]

                # Loop through all target gids and append spikes to appropriate array
                for target_gid in target_gid_list:
                    target_type = conn['target_type']
                    receptor = conn['receptor']
                    loc = conn['loc']
                    # Positions assigned based on gid in real network
                    target_pos_list = list(net.pos_dict[target_type])
                    if target_type in net.cell_types:
                        target_pos = target_pos_list[target_gid - list(net.gid_ranges[target_type])[0]]
                    else:
                        target_pos = target_pos_list[0]

                    # Get distance dependent weight/delay for connection
                    weight, delay = _get_gaussian_connection(
                        src_pos, target_pos, conn['nc_dict'], inplane_distance=net._inplane_distance)

                    delay_samples = int(round(delay / self.dt))
                    # if delay_samples == 0:
                    #     delay_samples = 1 # Can only deliver a spike at least 1 time step in the future

                    # Add delay to delay matrix
                    self.delay_matrix[src_gid, target_gid] = delay                    

                    # Delay spikes by delay_samples

                    if loc in net.cell_types[target_type].sect_loc:
                        sect_loc = net.cell_types[target_type].sect_loc[loc]
                    else:
                        sect_loc = [loc]
                        assert loc in net.cell_types[target_type].sections

                    for sec in sect_loc:
                        input_spike_name = f'{sec}_{receptor}'
                        input_sec_idx = self.neuron_info_dict[target_type]['isec_name_lookup'][input_spike_name]
                        self.connectivity_dict[target_type][target_gid - target_gid_list[0], input_sec_idx, src_gid] = weight

    def make_neuron_info(self):
        neuron_info_dict = dict()
        neuron_info_dict['L2_basket'] = {
            'vsec': ['soma'],
            'isec': ['soma_ampa', 'soma_gabaa', 'soma_nmda']}

        neuron_info_dict['L2_pyramidal'] = {
            'vsec': ['apical_trunk', 'apical_1', 'apical_tuft', 'apical_oblique', 'basal_1', 'basal_2', 'basal_3', 'soma'],
            'isec': ['apical_trunk_ampa', 'apical_trunk_nmda', 'apical_trunk_gabaa',
                    'apical_trunk_gabab', 'apical_1_ampa', 'apical_1_nmda', 'apical_1_gabaa',
                    'apical_1_gabab', 'apical_tuft_ampa', 'apical_tuft_nmda',
                    'apical_tuft_gabaa', 'apical_tuft_gabab', 'apical_oblique_ampa',
                    'apical_oblique_nmda', 'apical_oblique_gabaa', 'apical_oblique_gabab',
                    'basal_1_ampa', 'basal_1_nmda', 'basal_1_gabaa', 'basal_1_gabab',
                    'basal_2_ampa', 'basal_2_nmda', 'basal_2_gabaa', 'basal_2_gabab',
                    'basal_3_ampa', 'basal_3_nmda', 'basal_3_gabaa', 'basal_3_gabab',
                    'soma_gabaa', 'soma_gabab']}

        neuron_info_dict['L5_basket'] = {
            'vsec': ['soma'],
            'isec': ['soma_ampa', 'soma_gabaa', 'soma_nmda']}

        neuron_info_dict['L5_pyramidal'] = {
            'vsec': ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft', 'apical_oblique', 'basal_1', 'basal_2', 'basal_3', 'soma'],
            'isec': ['apical_trunk_ampa', 'apical_trunk_nmda', 'apical_trunk_gabaa',
                    'apical_trunk_gabab', 'apical_1_ampa', 'apical_1_nmda', 'apical_1_gabaa',
                    'apical_1_gabab', 'apical_2_ampa', 'apical_2_nmda', 'apical_2_gabaa',
                    'apical_2_gabab', 'apical_tuft_ampa', 'apical_tuft_nmda',
                    'apical_tuft_gabaa', 'apical_tuft_gabab', 'apical_oblique_ampa',
                    'apical_oblique_nmda', 'apical_oblique_gabaa', 'apical_oblique_gabab',
                    'basal_1_ampa', 'basal_1_nmda', 'basal_1_gabaa', 'basal_1_gabab',
                    'basal_2_ampa', 'basal_2_nmda', 'basal_2_gabaa', 'basal_2_gabab',
                    'basal_3_ampa', 'basal_3_nmda', 'basal_3_gabaa', 'basal_3_gabab',
                    'soma_gabaa', 'soma_gabab']
        }

        return neuron_info_dict


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    # scale = 1000.0 # controls steepness of surrogate gradient
    scale = 1e6 # controls steepness of surrogate gradient


    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply


#LSTM/GRU architecture for decoding
class model_network(nn.Module):
    def __init__(self, net, model_dict, dataset_dict, network_data, device='cuda:0', bidirectional=False):
        super(model_network, self).__init__()
        self.net = net
        self.device = device

        self.L5pyr_model: torch.Module = model_dict['L5_pyramidal']
        self.L2pyr_model: torch.Module = model_dict['L2_pyramidal']
        self.L5basket_model: torch.Module = model_dict['L5_basket']
        self.L2basket_model: torch.Module = model_dict['L2_basket']   
        self.model_dict: torch.ModuleDict[str, torch.Module] = nn.ModuleDict(model_dict)
        self.gid_ranges: Dict[str, np.ndarray] = {cell_type: list(gid_range) for cell_type, gid_range in net.gid_ranges.items()}

        self.kernel_size = model_dict['L5_pyramidal'].kernel_size
        self.dataset_dict = dataset_dict
        self.network_data = network_data
        connectivity_dict = network_data.connectivity_dict.copy()
        self.connectivity_dict = nn.ParameterDict(connectivity_dict)

        EI_dict = dict()
        for cell_type, conn in self.connectivity_dict.items():
            self.connectivity_dict[cell_type] = torch.from_numpy(conn).float().to(device)
            # EI_dict[cell_type] = torch.tensor(0.0).requires_grad_(True).to(device)
            for target_type, _ in self.connectivity_dict.items():
                EI_dict[f'{cell_type}_{target_type}'] = torch.tensor(0.0).requires_grad_(True).to(device)
        self.EI_dict = nn.ParameterDict(EI_dict)

        self.delay_matrix = network_data.delay_matrix

        self.scaler_dict: Dict[str, Dict[str, torch.Tensor]] = self.get_spike_scaler()

        self.soma_idx = 0
        self.threshold_dict = self.get_thresholds()
        self.threshold_dict['L5_basket'] = torch.tensor(50.0).to(device)
        self.threshold_dict['L2_basket'] = torch.tensor(50.0).to(device)
        self.threshold_dict['L5_pyramidal'] = torch.tensor(100.0).to(device)
        # self.threshold_dict['L2_pyramidal'] = torch.tensor(50.0).to(device)



    @torch.jit.export
    def scale_spikes(self, input_spikes: torch.Tensor, cell_type: str) -> torch.Tensor:
        # input_spikes *= (10 ** self.EI_dict[cell_type])
        input_spikes *= self.scaler_dict[cell_type]['spike_scale']
        input_spikes += self.scaler_dict[cell_type]['spike_min']
        return input_spikes

    def get_thresholds(self):
        threshold_dict = dict()
        for cell_type in self.net.cell_types:
            threshold = (self.net.threshold - self.scaler_dict[cell_type]['vsec_mean'][self.soma_idx]) / \
                self.scaler_dict[cell_type]['vsec_scale'][self.soma_idx]
            threshold_dict[cell_type] = threshold

        return threshold_dict

    def forward(self, L5pyr_spikes: torch.Tensor, L2pyr_spikes: torch.Tensor,
                L5basket_spikes: torch.Tensor, L2basket_spikes: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        input_spikes_dict: Dict[str, torch.Tensor] = {'L5_pyramidal': L5pyr_spikes, 'L2_pyramidal': L2pyr_spikes,
                            'L5_basket': L5basket_spikes, 'L2_basket': L2basket_spikes}

        cell_names = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket', 'L2_basket']

        h0_L5basket = torch.zeros(self.L5basket_model.n_layers, L5basket_spikes.size(0), self.L5basket_model.hidden_dim).to(self.device)
        c0_L5basket = h0_L5basket.clone()

        h0_L2basket = torch.zeros(self.L2basket_model.n_layers, L2basket_spikes.size(0), self.L2basket_model.hidden_dim).to(self.device)
        c0_L2basket = h0_L2basket.clone()

        h0_L5pyr = torch.zeros(self.L5pyr_model.n_layers, L5pyr_spikes.size(0), self.L5pyr_model.hidden_dim).to(self.device)
        c0_L5pyr = h0_L5pyr.clone()

        h0_L2pyr = torch.zeros(self.L2pyr_model.n_layers, L2pyr_spikes.size(0), self.L2pyr_model.hidden_dim).to(self.device)
        c0_L2pyr = h0_L2pyr.clone()
        
        pred_y_dict = {cell_type:
            [torch.zeros((input_spikes_dict[cell_type][:,0,:].size(0), self.model_dict[cell_type].output_size)).to(self.device),
             torch.zeros((input_spikes_dict[cell_type][:,0,:].size(0), self.model_dict[cell_type].output_size)).to(self.device)] for
                cell_type in cell_names}
        for time_idx in range(self.kernel_size, input_spikes_dict['L5_basket'].size(1)-1):
 
            batch_x = input_spikes_dict['L2_basket'][:, time_idx-self.L2pyr_model.kernel_size:time_idx, :].clone()
            out_L2_basket = self.scale_spikes(batch_x, 'L2_basket')

            batch_x = input_spikes_dict['L2_pyramidal'][:, time_idx-self.L2pyr_model.kernel_size:time_idx, :].clone()
            out_L2_pyr = self.scale_spikes(batch_x, 'L2_pyramidal')

            batch_x = input_spikes_dict['L5_basket'][:, time_idx-self.L5pyr_model.kernel_size:time_idx, :].clone()
            out_L5_basket = self.scale_spikes(batch_x, 'L5_basket')     

            batch_x = input_spikes_dict['L5_pyramidal'][:, time_idx-self.L5pyr_model.kernel_size:time_idx, :].clone()
            out_L5_pyr = self.scale_spikes(batch_x, 'L5_pyramidal')

            futures : List[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
            futures.append(torch.jit.fork(self.L2basket_model, out_L2_basket, h0_L2basket, c0_L2basket))
            futures.append(torch.jit.fork(self.L2pyr_model, out_L2_pyr, h0_L2pyr, c0_L2pyr))
            futures.append(torch.jit.fork(self.L5basket_model, out_L5_basket, h0_L5basket, c0_L5basket ))
            futures.append(torch.jit.fork(self.L5pyr_model, out_L5_pyr, h0_L5pyr, c0_L5pyr))

            # Collect the results from the launched tasks
            results :  List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for future in futures:
                results.append(torch.jit.wait(future))

            pred_y_dict['L2_basket'].append(results[0][0][:,-1, :])
            pred_y_dict['L2_pyramidal'].append(results[1][0][:,-1, :])
            pred_y_dict['L5_basket'].append(results[2][0][:,-1, :])
            pred_y_dict['L5_pyramidal'].append(results[3][0][:,-1, :])

            h0_L2basket, c0_L2basket = results[0][1], results[0][2]
            h0_L2pyr, c0_L2pyr = results[1][1], results[1][2]
            h0_L5basket, c0_L5basket = results[2][1], results[2][2]
            h0_L5pyr, c0_L5pyr = results[3][1], results[3][2]

            # Detect spikes and update input spikes
            for cell_type in cell_names:
                pred_t1 = pred_y_dict[cell_type][-1][:, self.soma_idx]
                pred_t2 = pred_y_dict[cell_type][-2][:, self.soma_idx]

                spike_greater = spike_fn(pred_t1 - self.threshold_dict[cell_type])
                spike_less = spike_fn(pred_t2 - self.threshold_dict[cell_type])

                spike_mask = spike_greater * (1 - spike_less)
                            
                for target_type in cell_names:
                    input_spikes_dict[target_type][:, time_idx+1, :] += \
                        torch.matmul(self.connectivity_dict[target_type][:, :, self.gid_ranges[cell_type]], \
                            spike_mask * (10 ** self.EI_dict[f'{cell_type}_{target_type}']))

        return (torch.stack(pred_y_dict['L2_basket']), torch.stack(pred_y_dict['L2_pyramidal']),
                torch.stack(pred_y_dict['L5_basket']), torch.stack(pred_y_dict['L5_pyramidal']))
    
    def get_spike_scaler(self):
        scaler_dict = dict()
        for cell_type in self.net.cell_types:
            scaler_dict[cell_type] = {
                'spike_min': torch.tensor(self.dataset_dict[cell_type].datasets[0].input_spike_scaler.min_).float().to(self.device),
                'spike_scale': torch.tensor(self.dataset_dict[cell_type].datasets[0].input_spike_scaler.scale_).float().to(self.device),
                'vsec_mean': torch.tensor(self.dataset_dict[cell_type].datasets[0].vsec_scaler.mean_).float().to(self.device),
                'vsec_scale': torch.tensor(self.dataset_dict[cell_type].datasets[0].vsec_scaler.scale_).float().to(self.device),
            }

        return scaler_dict

class ConcatTensorDataset(torch.utils.data.ConcatDataset):
    r"""ConcatDataset of TensorDatasets which supports getting slices and index lists/arrays.
    This dataset allows the use of slices, e.g. ds[2:4] and of arrays or lists of multiple indices
    if all concatenated datasets are either TensorDatasets or Subset or other ConcatTensorDataset instances
    which eventually contain only TensorDataset instances. If no slicing is needed,
    this class works exactly like torch.utils.data.ConcatDataset and can concatenate arbitrary
    (not just TensorDataset) datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    def __init__(self, datasets):
        super(ConcatTensorDataset, self).__init__(datasets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in range(self.__len__())[idx]]
            return tuple(map(torch.stack, zip(*rows)))
        elif isinstance(idx, (list, np.ndarray)):
            rows = [super(ConcatTensorDataset, self).__getitem__(i) for i in idx]
            return tuple(map(torch.stack, zip(*rows)))
        else:
            return super(ConcatTensorDataset, self).__getitem__(idx)