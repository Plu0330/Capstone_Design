import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
import math
import time

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
        
"""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

# p1 = [100.0,
#  20.0,
#  25.0,
#  30.0,
#  35.0,
#  40.0,
#  45.0,
#  50.0,
#  55.0,
#  60.0,
#  65.0,
#  70.0,
#  75.0,
#  80.0,
#  85.0,
#  90.0,
#  95.0,
#  99.9,
#  100.0]

p1 = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]

#p1 = ['99.9']

def validate(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])
    
    use_layer = ['layer4.1.conv2', 'layer4.0.conv1', 'layer1.1.conv1', 'layer3.0.conv2']

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    for p in p1:
        print(f'{p}% validate start')
        epoch_start_time = time.time()
        acc_cum = 0.0
        batch_size = configuration['val_dataset_params']['loader_params']['batch_size']

        for i, data in enumerate(val_dataset):
            print('{} / {}'.format(i,math.floor(val_dataset_size/batch_size)))
            activation = {}

            model.set_input(data)
            for name,module in model.netsign.named_modules():
                if name in use_layer:
                    module.register_forward_hook(get_activation(name))

            model.test()
            acc = model.post_epoch_callback_loss(activation = activation, p = float(p))
            acc_cum += acc

        acc_res = acc_cum / (i+1)
        print('This epoch\' accuracy : {0}'.format(acc_res))

        save_path = 'C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\results\\experiment\\FeatureBag\\layerSelectAlgorithm\\baseline'
        save_name = f'rsenet18+DB(30~40%)'
        model.df_save(path=os.path.join(save_path, save_name+'.csv'),name = save_name, acc = acc_res)
        print(f'{p}% \t Time Taken: {time.time() - epoch_start_time} sec')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    #parser.add_argument('configfile', help='path to the configfile')

    activation = {}
    args = parser.parse_args()
    validate(config_file='C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\config_sign.json')
