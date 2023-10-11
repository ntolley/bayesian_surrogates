import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
import multiprocessing
from joblib import Parallel, delayed
import pickle


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
            # train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
            train_loss = criterion(output[:,:,:], batch_y[:,:,:])

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
                validation_loss = criterion(output[:,:,:], batch_y[:,:,:])

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
def train_validate_test_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, testing_generator,device, print_freq=10, early_stop=20):
    train_loss_array = []
    validation_loss_array = []
    test_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        test_batch_loss = []
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
            #Generate validation set predictions
            for batch_x, batch_y in validation_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

                validation_batch_loss.append(validation_loss.item())

            validation_loss_array.append(validation_batch_loss)

            #Generate test set predictions
            for batch_x, batch_y in testing_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                test_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

                test_batch_loss.append(test_loss.item())

            test_loss_array.append(test_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)
        test_epoch_loss = np.mean(test_batch_loss)
        test_epoch_std = np.std(test_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            min_test_loss = np.copy(test_epoch_loss)
            min_test_std = np.copy(test_epoch_std)


        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.2f}  ... Validation Loss: {:.2f} ... Test Loss: {:.2f}'.format(train_epoch_loss, validation_epoch_loss, test_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'min_test_loss':min_test_loss, 'min_test_std':min_test_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'test_loss_array':test_loss_array, 'max_epochs':max_epochs}
    return loss_dict

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
    def __init__(self, network_dataset, cell_type='L5_pyramidal', data_step_size=1,
                 window_size=100, input_spike_scaler=None, vsec_scaler=None, isec_scaler=None,
                 device='cpu'):
        
        self.cell_type = cell_type
        self.num_cells = len(network_dataset.net.gid_ranges[self.cell_type])
        self.data_step_size = data_step_size
        self.window_size = window_size
        self.device = device

        self.input_spike_list, self.vsec_list, self.isec_list = self.process_data(network_dataset)
        assert len(self.input_spike_list) == len(self.vsec_list) == len(self.isec_list) == self.num_cells

        if input_spike_scaler is None:
            self.input_spike_scaler = StandardScaler()
            self.input_spike_scaler.fit(np.vstack(self.input_spike_list))
        
        if vsec_scaler is None:
            self.vsec_scaler = StandardScaler()
            self.vsec_scaler.fit(np.vstack(self.vsec_list))
        
        if isec_scaler is None:
            self.isec_scaler = StandardScaler()
            self.isec_scaler.fit(np.vstack(self.isec_list))
 

        self.input_spike_unfolded, self.vsec_unfolded, self.isec_unfolded = self.unfold_data()
        
        # X is one step behind y
        self.X_tensor = self.input_spike_unfolded[:, :-1, :]
        self.y_tensor = self.vsec_unfolded[:, 1:, :]
        assert self.X_tensor.shape[0] == self.y_tensor.shape[0]
        self.num_samples = self.X_tensor.shape[0]

        self.X_tensor = self.X_tensor.to(self.device)
        self.y_tensor = self.y_tensor.to(self.device)

    
    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        return self.X_tensor[slice_index,:,:], self.y_tensor[slice_index,:,:]
    

    def process_data(self, network_dataset):
        gid_list = network_dataset.net.gid_ranges[self.cell_type]
        input_spike_list, vsec_list, isec_list = list(), list(), list()
        for gid in gid_list:
            input_spike_list.append(network_dataset.input_spike_dict[gid].T)
            vsec_list.append(network_dataset.neuron_data_dict[gid].vsec_array.T)
            isec_list.append(network_dataset.neuron_data_dict[gid].isec_array.T)
        
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
        
        