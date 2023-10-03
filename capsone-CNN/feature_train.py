import argparse
import torch
from utils import parse_configuration, Activation
from datasets import create_dataset
from models import create_model
from models.resnet import resnet101, resnet50, resnet18, resnet34
#from models.resnet_custom_model import layer_list
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import json
import time
from utils import Generate_noise_featureMap
import os
import sys
import numpy as np
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


def feature_train(config_file, export = False):

    print(torch.cuda.is_available())
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    feature_dataset = create_dataset(configuration['feature_dataset_params'])
    feature_dataset_size = len(feature_dataset)
    print('The number of dataset size = {0}'.format(feature_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])


    feature_epochs = configuration['model_params']['feature_epochs']
    feature_db_path = configuration['model_params']['db_path']
    class_num = configuration['feature_dataset_params']['class_num']

    activation = {}
    # generate feature map

    use_layer = ['layer4.1.conv2', 'layer4.0.conv1', 'layer1.1.conv1', 'layer3.0.conv2']

    for c in range(class_num):
        dirpath = os.path.join(feature_db_path,str(c))
        if os.path.isdir(dirpath) == False:
            os.mkdir(dirpath)

    # for feature map training
    for epoch in range(feature_epochs):
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        for name,module in model.netsign.named_modules():
            #if isinstance(module,nn.Conv2d):
            if name in use_layer:
                module.register_forward_hook(get_activation(name))
        
        epoch_start_time = time.time()
        feature_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        batch_size = configuration['feature_dataset_params']['loader_params']['batch_size']
        
        for i, data in enumerate(feature_dataset):
            print('{} / {}'.format(i,math.floor(feature_dataset_size/batch_size)))
            
            activation = {}
            model.set_input(data)
            
            #print('label : ', model.label)
            #print('input : ', model.input)
            #print('input : ', type(str(model.label)))
        
            model.feature_test()
            model.mean_feature(activation)

            #model.optimize_feature(feature_db_path,activation)
            

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch,
              feature_epochs, time.time() - epoch_start_time))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Get feature map.')

    arg = parser.parse_args()
    activation = {}
    feature_train(config_file = 'C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\config_sign.json')