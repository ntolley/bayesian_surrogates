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
        self.neuron_data_dict = dict()
        self.input_spike_dict = dict()
        
        for cell_type, gid_list in net.gid_ranges.items():
            for gid in gid_list:
                self.neuron_data_dict[gid] = SingleNeuron_Data(net, gid, soma_filter=soma_filter)

                # Initialize blank arrays for spikes recieved by each cell
                if cell_type in net.cell_types:
                    self.input_spike_dict[gid] = np.zeros(self.neuron_data_dict[gid].isec_array.shape)

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
                # **TODO** - need to add delay calculation
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

                    conn_spikes *= weight

                    if loc in net.cell_types[target_type].sect_loc:
                        sect_loc = net.cell_types[target_type].sect_loc[loc]
                    else:
                        sect_loc = [loc]
                        assert loc in net.cell_types[target_type].sections

                    for sec in sect_loc:
                        input_spike_name = f'{sec}_{receptor}'
                        input_spike_idx = self.neuron_data_dict[target_gid].isec_name_lookup[input_spike_name]
                        self.input_spike_dict[target_gid][input_spike_idx, :] += conn_spikes
            

class CellType_Dataset_Fast(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, net, cell_type='L5_pyramidal', data_step_size=1,
                 window_size=100, input_spike_scaler=None, vsec_scaler=None, isec_scaler=None,
                 soma_filter=False, device='cpu'):
        
        network_data = Network_Data(net, soma_filter=soma_filter)
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
            # self.input_spike_scaler = MinMaxScaler(feature_range=(0, 100))
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

def beta_tuning_param_function(net, theta_dict):
    conn_type_list = {'EI_connections': 'EI', 'EE_connections': 'EE', 
                      'II_connections': 'II', 'IE_connections': 'IE'}
    
    seed_rng = np.random.default_rng(theta_dict['theta_extra']['sample_idx'])
    seed_array = seed_rng.integers(10e5, size=100)

    seed_count = 0
    for conn_type_name, conn_suffix in conn_type_list.items():
        conn_prob_name = f'{conn_suffix}_prob'
        conn_gscale_name = f'{conn_suffix}_gscale'
        
        conn_indices = theta_dict['theta_extra'][conn_type_name]
        probability = theta_dict[conn_prob_name]
        gscale = theta_dict[conn_gscale_name]
        
        for conn_idx in conn_indices:
            # Prune connections using internal connection_probability function
            _connection_probability(
                net.connectivity[conn_idx], probability=probability, conn_seed=seed_array[seed_count])
            net.connectivity[conn_idx]['probability'] = probability
            net.connectivity[conn_idx]['nc_dict']['A_weight'] *= gscale
            seed_count = seed_count + 1

    for conn_idx in range(len(net.connectivity)):
        net.connectivity[conn_idx]['nc_dict']['lamtha'] = theta_dict['theta_extra']['lamtha']          
        
    rate = 10
    # Add Poisson drives
    weights_ampa_d1 = {'L2_pyramidal': theta_dict['L2e_distal'], 'L5_pyramidal': theta_dict['L5e_distal'],
                       'L2_basket': theta_dict['L2i_distal']}
    rates_d1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate}

    net.add_poisson_drive(
        name='distal', tstart=0, tstop=None, rate_constant=rates_d1, location='distal', n_drive_cells='n_cells',
        cell_specific=True, weights_ampa=weights_ampa_d1, weights_nmda=None, space_constant=1e50,
        synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-1], conn_seed=seed_array[-2])

    weights_ampa_p1 = {'L2_pyramidal': theta_dict['L2e_proximal'], 'L5_pyramidal': theta_dict['L5e_proximal'],
                       'L2_basket': theta_dict['L2i_proximal'], 'L5_basket': theta_dict['L5i_proximal']}
    rates_p1 = {'L2_pyramidal': rate, 'L5_pyramidal': rate, 'L2_basket': rate, 'L5_basket': rate}

    net.add_poisson_drive(
        name='proximal', tstart=0, tstop=None, rate_constant=rates_p1, location='proximal', n_drive_cells='n_cells',
        cell_specific=True, weights_ampa=weights_ampa_p1, weights_nmda=None, space_constant=1e50,
        synaptic_delays=0.0, probability=1.0, event_seed=seed_array[-3], conn_seed=seed_array[-4])