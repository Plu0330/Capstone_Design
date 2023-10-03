import json
import math
import numpy as np
import os
from pathlib import Path
import torch
from torch.optim import lr_scheduler
from itertools import zip_longest


def transfer_to_device(x, device):
    """Transfers pytorch tensors or lists of tensors to GPU. This
        function is recursive to be able to deal with lists of lists.
    """
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to_device(x[i], device)
    else:
        x = x.to(device)
    return x


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file


def get_scheduler(optimizer, configuration, last_epoch=-1):
    """Return a learning rate scheduler.
    """
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=configuration['lr_decay_iters'], gamma=0.3, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [{0}] is not implemented'.format(configuration['lr_policy']))
    return scheduler


def stack_all(list, dim=0):
    """Stack all iterables of torch tensors in a list (i.e. [[(tensor), (tensor)], [(tensor), (tensor)]])
    """
    return [torch.stack(s, dim) for s in list]

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Activation():
    def __init__(self,module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data.numpy()

    def remove(self):
        self.hook.remove()


class Generate_noise_featureMap():
    def __init__(self, configuration,activation_info):
        self.num = configuration['target_dataset_params']['class_num']
        self.activation_list = activation_info
        self.activation_path = configuration['model_params']['db_path']
    def generate(self):
        for i in range(self.num):
            path = os.path.join(self.activation_path,str(i))
            if os.path.isdir(path) == False:
                os.mkdir(path)
            for s in self.activation_list.items():
                file_name = f'{i}_{s[0]}'
                #print('s0 : ',s[0])
                #print('s1 : ', s[1])
                #print('s0 type : ',type(s[0]))
                #print('s1 type : ', type(s[1]))
                #print('s1"size : ', s[1].shape)
                temp_np = torch.rand(s[1].shape)
                np.save(os.path.join(path,file_name),temp_np)
        
        print('noise feature map created')

    def concat_features(self):
        for i in range(self.num):
            file_name = f'{i}_activation'
            path = os.path.join(self.activation_path,str(i))
            arrays = []
            for f in os.listdir(path):
                arrays.append(np.load(os.path.join(path,f)))
            shapes = [n.shape for n in arrays]
            final = np.vstack(shapes).max(axis=0)
            out = np.concatenate([np.pad(a, list(zip_longest([0], final-a.shape,
                                                 fillvalue=0)), 'constant')
                      for a in arrays])
            np.save(os.path.join(file_name), out)
        print('concat complete')

