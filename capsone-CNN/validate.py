import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from sklearn.metrics import accuracy_score
import json
import numpy as np
import torch

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
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
    cnn_predicts = json.load(open(r'C:\Users\ms9804\Desktop\capstone\5.capstoneTestVer1\for_merge\cnn_predictions.json', encoding = 'utf8'))
    cnn_labels = json.load(open(r'C:\Users\ms9804\Desktop\capstone\5.capstoneTestVer1\for_merge\cnn_labels.json', encoding = 'utf8'))
    loss_cum = 0.0
    acc_cum = 0.0

    for i, data in enumerate(val_dataset):
        model.set_input(data)
        model.test()
        acc,loss = model.post_epoch_callback_loss(configuration['model_params']['load_checkpoint'])
        loss_cum += loss
        acc_cum += acc
        #print('acc : {}, loss : {}'.format(acc_cum, loss_cum))
        
    loss_res = loss_cum / (i+1)
    acc_res = acc_cum / (i+1)
    #print('This epoch\' accuracy : {0}, loss : {1}'.format(acc_res,loss_res))
    print('This epoch\' accuracy : {0}, loss : {1}'.format(acc_res,loss_res))
    
    save_path = 'C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\results'
    save_name = 'cnn using r2v'
    model.df_save(path=os.path.join(save_path, save_name+'.csv'),name = save_name, acc = acc_res, loss = loss_res)
    model.save_info()
    merged = model.change_predict()
    
    acc = accuracy_score(merged,cnn_labels)
    print('This epoch\' accuracy : {0}'.format(acc))
    
    #print(merged)
    # acc = 0
    # for i in range(len(merged)):
    #     if merged[i] == cnn_predicts[i]:
    #         acc+=1
    # acc = acc / len(merged)

    # print('This epoch\' accuracy : {0}'.format(acc))
    # print(f'ori prediction value : {cnn_predicts[1]}, type : {type({cnn_predicts[1]})}')
    # print(f'ori labels value : {cnn_labels[1]}, type : {type({cnn_labels[1]})}')
    # print(f'changed prediction value : {merged[1]}, type : {type({merged[1]})}')
    # tt = torch.Tensor(merged)
    # print(f'tensor value : {tt[1]}, type : {type({tt[1]})}')
    # print(type(tt))
    
    
    #print(merged)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    #parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(config_file='C:\\Users\\ms9804\\Desktop\\capstone\\5.capstoneTestVer1\\config_sign.json')