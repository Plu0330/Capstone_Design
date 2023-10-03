from utils.visualizer import Visualizer
import time
from models import create_model
import math
from utils import parse_configuration
from datasets import create_dataset
import argparse
import torch
import gc

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""


def train(config_file, export=True):
    print(torch.cuda.is_available())
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    train_dataset = create_dataset(configuration['train_dataset_params'])
    #dataset 내부 확인
    #print(len(train_dataset))
    #print(train_dataset)
    #print(size(train_dataset))
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))
    
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    if(configuration['model_params']['load_checkpoint'] != -1):
        model.load_networks(configuration['model_params']['load_checkpoint'])
        model.load_optimizers(configuration['model_params']['load_checkpoint'])

    print('Initializing visualization...')
    # create a visualizer that displays images and plots
    visualizer = Visualizer(configuration['visualization_params'])

    starting_epoch = configuration['model_params']['load_checkpoint'] + 1
    num_epochs = configuration['model_params']['max_epochs']

    save_path = 'C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\results'
    save_name = 'random_erasing(p=0.5)'

    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
        #eval_batch_size = configuration['val_dataset_params']['loader_params']['batch_size']

        model.train()
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            visualizer.reset()
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            
            model.forward()
            model.backward()

            if i % configuration['model_update_freq'] == 0:
                # calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

            if i % configuration['printout_freq'] == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, num_epochs, i, math.floor(
                    train_iterations / train_batch_size), losses)
                visualizer.plot_current_losses(epoch, float(
                    i) / math.floor(train_iterations / train_batch_size), losses)   

            # if i % configuration['save_freq'] == 0:
            #     acc_epoch,loss_epoch = model.get_acc_loss(configuration['model_params']['load_checkpoint'])
            #     model.df_save_epoch(path=os.path.join(save_path, save_name+'.csv'),epoch = i, acc = acc_epoch, loss = loss_epoch)
        
        loss_cum = 0.0
        acc_cum = 0.0
        model.eval()

        for i, data in enumerate(val_dataset):
            model.set_input(data)
            model.test()
            acc,loss = model.post_epoch_callback_loss(configuration['model_params']['load_checkpoint'])
            loss_cum += loss
            acc_cum += acc
        
        loss_res = loss_cum / (i+1)
        acc_res = acc_cum / (i+1)
        print('This epoch\'s accuracy : {0}, loss : {1}'.format(acc_res,loss_res))

        if epoch % configuration['save_freq'] == 0:
            model.df_save_epoch(path=os.path.join(save_path, save_name+'.csv'),epoch = epoch, acc = acc_res, loss = loss_res)

        if epoch == (configuration['model_params']['max_epochs'] - 1):
            model.df_save_epoch(path=os.path.join(save_path, save_name+'.csv'),epoch = (configuration['model_params']['max_epochs'] - 1), acc = acc_res, loss = loss_res)
        
        # total_loss = 0
        # for i, data in enumerate(val_dataset):
        #         model.set_input(data)
        #         model.test()     
                    

        print('complete validation')
        # model.post_epoch_callback(epoch, visualizer)
        train_dataset.dataset.post_epoch_callback(epoch)

        print('Saving model at the end of epoch {0}'.format(epoch))
        model.save_networks(epoch)
        model.save_optimizers(epoch)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch,
              num_epochs, time.time() - epoch_start_time))

        model.update_learning_rate()  # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_configuration = configuration['train_dataset_params']
        # set batch size to 1 for tracing
        custom_configuration['loader_params']['batch_size'] = 1
        dl = train_dataset.get_custom_dataloader(custom_configuration)
        sample_input = next(iter(dl))  # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    #parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    # train(args.configfile)
    train(config_file='C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\config_sign.json')
