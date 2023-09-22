import os
import numpy as np
import vtk
from vedo import *
from src.train.step2_get_list import SplitData






class DataAugmentation:
    def __init__(self):
        pass


    def GetVTKTransformationMatrix(self,rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
        '''
        get transformation matrix (4*4)
        return: vtkMatrix4x4
        '''
        Trans = vtk.vtkTransform()

        ry_flag = np.random.randint(0,2) #if 0, no rotate
        rx_flag = np.random.randint(0,2) #if 0, no rotate
        rz_flag = np.random.randint(0,2) #if 0, no rotate
        if ry_flag == 1:
            # rotate along Yth axis
            Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
        if rx_flag == 1:
            # rotate along Xth axis
            Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
        if rz_flag == 1:
            # rotate along Zth axis
            Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

        trans_flag = np.random.randint(0,2) #if 0, no translate
        if trans_flag == 1:
            Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                            np.random.uniform(translate_Y[0], translate_Y[1]),
                            np.random.uniform(translate_Z[0], translate_Z[1])])

        scale_flag = np.random.randint(0,2)
        if scale_flag == 1:
            Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                        np.random.uniform(scale_Y[0], scale_Y[1]),
                        np.random.uniform(scale_Z[0], scale_Z[1])])

        matrix = Trans.GetMatrix()
        return matrix
    



    def initiate_data_augmentation(self,ip_path,op_save_path):
        num_augmentations = 20
        i_sample=0
        for file_name in os.listdir(ip_path):
            i_sample= i_sample + 1
            for i_aug in range(num_augmentations):

                
                output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
                vtk_matrix = self.GetVTKTransformationMatrix() #use default random setting
                mesh = load(os.path.join(ip_path, file_name))
                mesh.apply_transform(vtk_matrix)
                io.write(mesh, os.path.join(op_save_path, output_file_name))

    


    

if __name__ == "__main__":
    ip_path = "./src/train/ip_vtk"
    op_save_path='./src/train/augmentation_vtk_data'
    dt= DataAugmentation()
    dt.initiate_data_augmentation(ip_path,op_save_path)

    data_path = './src/train/augmentation_vtk_data/'
    output_path = './src/train/'
    train_size = 0.8
    s=SplitData()
    s.get_list(data_path, output_path, train_size)

 






















