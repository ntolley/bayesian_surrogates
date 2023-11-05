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
from neurodsp.spectral import compute_spectrum, trim_spectrum
import scipy
import utils
import multiprocessing
import dill
device = torch.device("cuda:0")
num_cores = multiprocessing.cpu_count()
torch.backends.cudnn.enabled = True

def train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, print_freq=10, early_stop=20):
    kernel_size = model.kernel_size

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

            output_sequence = []
            h0 = torch.zeros(model.n_layers, batch_x.size(0), model.hidden_dim).to(device)
            c0 = torch.zeros(model.n_layers, batch_x.size(0), model.hidden_dim).to(device)

            for t in range(kernel_size, batch_x.size(1)-1):
                output, h0, c0 = model(batch_x[:,(t-kernel_size):t, :], h0, c0)
                output_sequence.append(output)

            output_sequence = torch.cat(output_sequence, dim=1)
            train_loss = criterion(output_sequence, batch_y[:,kernel_size+1:,:])

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

                output_sequence = []
                h0 = torch.zeros(model.n_layers, batch_x.size(0), model.hidden_dim).to(device)
                c0 = torch.zeros(model.n_layers, batch_x.size(0), model.hidden_dim).to(device)
                for t in range(kernel_size, batch_x.size(1)):
                    output, h0, c0 = model(batch_x[:,(t-kernel_size):t, :], h0, c0)
                    output_sequence.append(output)

                output_sequence = torch.cat(output_sequence, dim=1)
                validation_loss = criterion(output_sequence, batch_y[:,kernel_size:,:])

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


dataset_type_list = ['subthreshold', 'suprathreshold', 'connected']
# dataset_type_list = ['connected']


cell_type_list = ['L5_pyramidal', 'L2_pyramidal', 'L5_basket', 'L2_basket']


for dataset_type in dataset_type_list:
    # dipole_array = np.array(
    #         [np.load(f'datasets_{dataset_type}/dipole_data/dipole_{sample_idx}.npy') for
    #          sample_idx in range(100)])
    for cell_type in cell_type_list:
        print('\n')
        print(f'___Training {dataset_type} {cell_type} model___')

        # # Ensure simulations are subthreshold
        # if dataset_type == 'subthreshold':
        #     # Threshold for detecting spikes in dipole may need to be hand tuned based on number of cells
        #     sim_indices = np.where(dipole_array.max(axis=1) < 2e-3)[0]
        #     len(sim_indices)
        # else:
        sim_indices = np.arange(100)

        # Network size for different cell types
        if cell_type == 'L2_basket' or cell_type == 'L5_basket':
            hidden_dim = 8
            n_layers = 2
        else:
            hidden_dim = 64
            n_layers = 5


        # Set up training and validation datasets
        num_sims = len(sim_indices)
        num_train = int(num_sims * 0.8)
        num_validation = num_sims - num_train

        training_indices = sim_indices[0:num_train]
        validation_indices = sim_indices[num_train:]


        training_set = torch.utils.data.ConcatDataset(
            [torch.load(f'datasets_{dataset_type}/training_data/{cell_type}_dataset_{idx}.pt') for
             idx in training_indices])
        validation_set = torch.utils.data.ConcatDataset(
            [torch.load(f'datasets_{dataset_type}/training_data/{cell_type}_dataset_{idx}.pt') for
             idx in validation_indices])

        _, input_size = training_set[0][0].detach().cpu().numpy().shape
        _, output_size = training_set[0][1].detach().cpu().numpy().shape

        batch_size = 500
        num_cores = 8
        pin_memory = True

        train_params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory':pin_memory}
        train_eval_params = {'batch_size': batch_size, 'shuffle': False, 'pin_memory':pin_memory}
        validation_params = {'batch_size': batch_size, 'shuffle': True,  'pin_memory':pin_memory}
        test_params = {'batch_size': batch_size, 'shuffle': False, 'pin_memory':pin_memory}

        training_generator = torch.utils.data.DataLoader(training_set, **train_params)
        training_eval_generator = torch.utils.data.DataLoader(training_set, **train_eval_params)
        validation_generator = torch.utils.data.DataLoader(validation_set, **validation_params)

        validation_generator = torch.utils.data.DataLoader(validation_set, **test_params)


        # Initialize and train model
        model_pytorch = utils.model_celltype_lstm(input_size=input_size, output_size=output_size,
                                          hidden_dim=hidden_dim, n_layers=n_layers, device=device)
        model = torch.jit.script(model_pytorch).to(device)

        if dataset_type == 'suprathreshold':
            model.load_state_dict(torch.load(f'subthreshold_models/{cell_type}_subthreshold_model.pt'))
        if dataset_type == 'connected':
            model.load_state_dict(torch.load(f'suprathreshold_models/{cell_type}_suprathreshold_model.pt'))

        lr = 0.01
        weight_decay = 0
        max_epochs = 1000
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        #Train model
        loss_dict = train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, 5, 10)

        torch.save(model.state_dict(), f'{dataset_type}_models/{cell_type}_{dataset_type}_model.pt')
        with open(f'{dataset_type}_models/{cell_type}_{dataset_type}_loss_dict.pkl', 'wb') as f:
            dill.dump(loss_dict, f)

        del training_set, validation_set, training_generator, validation_generator, model_pytorch, model, loss_dict,
