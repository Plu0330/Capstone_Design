import argparse
import torch
from utils import parse_configuration, Activation
from datasets import create_dataset
from models import create_model
from models.resnet import resnet101, resnet50, resnet18, resnet34
import torch.nn as nn
import json
import time
from utils import Generate_noise_featureMap
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

def feature_validate(config_file, export = False):
    epoch_start_time = time.time()
    print(torch.cuda.is_available())
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of dataset size = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])


    feature_epochs = configuration['model_params']['feature_epochs']
    feature_db_path = configuration['model_params']['db_path']

    use_layer = ['layer4.1.conv2', 'layer4.0.conv1', 'layer1.1.conv1', 'layer3.0.conv2']
    
    #feature_db_path = configuration['model_params']['test_db_path']

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    accuracy = 0.0
    batch_size = configuration['val_dataset_params']['loader_params']['batch_size']
    
    
    # for feature map validate
    for i, data in enumerate(val_dataset):
        print('{} / {}'.format(i,math.floor(val_dataset_size/batch_size)))
        activation = {}
        model.set_input(data)
        
        for name,module in model.netsign.named_modules():
            if name in use_layer:
                module.register_forward_hook(get_activation(name))
        
        
        model.feature_test()

        accuracy += model.post_epoch_callback_feature_loss(activation)
        
    model.save_accuracy()
    print(f'Test Time Taken: {time.time() - epoch_start_time} sec')
    print(f'test accuracy : {accuracy/(i+1)}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Get feature map.')

    arg = parser.parse_args()
    activation = {}
    feature_validate(config_file = 'C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\config_sign.json')
    