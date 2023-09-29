import os
import numpy as np
import vtk
from vedo import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from src.mesh_dataset import *
from src.meshsegnet import *
from src.loss_and_metrics import *
import utils
import pandas as pd
from src.train.step2_get_list import SplitData
from src.train.step3_training import ModelTrainer
from src.train.step1_data_augmentation import DataAugmentation


if __name__ == "__main__":

    #Step1
    print("Starting Data Augmentation ")
    ip_path = "./src/train/ip_vtk"
    op_save_path='./src/train/augmentation_vtk_data'
    dt= DataAugmentation()
    dt.initiate_data_augmentation(ip_path,op_save_path)
    print('--------------------------------------------')
    print("Data Augmentation done")


    #Step2
    aug_data_path = op_save_path
    split_output_path = './src/train/train_test_splits/'
    train_size = 0.8
    s=SplitData()
    s.get_list(aug_data_path, split_output_path, train_size)
    print('--------------------------------------------')
    print("Train-test Splits done")

   
   #Step3
    train_path = './src/train/train_test_splits/train_list_1.csv' # use 1-fold as example
    val_path = './src/train/train_test_splits/val_list_1.csv' # use 1-fold as example
    model_op_path = './src/models'
    model_name = 'MeshSegNet_17_classes_60samples'
    t=ModelTrainer()
    t.initiate_model_trainer(train_path, val_path, model_op_path, model_name)