import os
import numpy as np
import torch
import torch.nn as nn
from src.meshsegnet import *
from vedo import *
import pandas as pd
from src.loss_and_metrics import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
from src.predict.predict_pipeline import PredictPipeline






if __name__=="__main__": 

    model_name ='MeshSegNet_17_classes_396_best.tar'
    ip_path = './src/predict/input/lower/'
    op_path = './src/predict/output/lower/'
    exctract_c= False
    p=PredictPipeline()



    

    for file_num in os.listdir(ip_path):
        if exctract_c== True:
            binary_mesh= p.predict_c(ip_path,file_num)
            cshape=p.get_cshape(binary_mesh,file_num)
            new_ip_path='./src/predict/extracted_c'
            p.predict_labels(model_name, new_ip_path, op_path, file_num)
            

        else:
            p.predict_labels(model_name, ip_path, op_path, file_num)



 
