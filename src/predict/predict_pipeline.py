import os
import numpy as np
import torch
import torch.nn as nn
from src.meshsegnet import *
import vedo
from vedo import *
import pandas as pd
from src.loss_and_metrics import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph


class PredictPipeline:
    def __init__(self):
        pass
    




    def predict_c(self, ip_path, file_num):

    
        gpu_id = 0
        #torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)
        upsampling_method = 'KNN'

        model_path = './src/models'
        model_name ="MeshSegNet_2_classes_80noflip_best.tar"

        mesh_path = ip_path # need to modify
        #sample_filenames = ['{}.stl'.format(i) for i in range(2,len(os.listdir(ip_path)) + 1)] # need to modify
        #i_sample='{}.stl'.format(file_num)
        i_sample=file_num
        print(i_sample)
        

        output_path = './src/predict/extracted_c'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        num_classes =2
        num_channels = 15

        # set model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

        # load trained model
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        model = model.to(device, dtype=torch.float)

        #cudnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


        # Predicting
        model.eval()
        with torch.no_grad():
            #for i_sample in sample_filenames:

            start_time = time.time()
            # create tmp folder
            tmp_path = './.tmp/'
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            print('Predicting Sample filename: {}'.format(i_sample))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, i_sample))

            # pre-processing: downsampling
            print('\tDownsampling...')
            target_num = 10000
            ratio = target_num/mesh.NCells() # calculate ratio
            mesh_d = mesh.clone()
            mesh_d.decimate(fraction=ratio)
            predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            cells = np.zeros([mesh_d.NCells(), 9], dtype='float32')
            for i in range(len(cells)):
                cells[i][0], cells[i][1], cells[i][2] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            original_cells_d = cells.copy()

            mean_cell_centers = mesh_d.centerOfMass()
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2)
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            mesh_d.celldata['Normal']=mesh_normals

            # preprae input
            points = mesh_d.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            barycenters = mesh_d.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            #mesh2 = mesh_d.clone()
            #mesh2.celldata['Label']=predicted_labels_d
            #vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            # refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

            # unaries
            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            # parawise
            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            #edges
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            cells = original_cells_d.copy()
            barycenters = mesh_d.cellCenters() # don't need to copy
            cell_ids = np.asarray(mesh_d.faces())

            lambda_c = 30
            edges = np.empty([1, 3], order='C')
            for i_node in range(cells.shape[0]):
                # Find neighbors
                nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
                nei_id = np.where(nei==2)
                for i_nei in nei_id[0][:]:
                    if i_node < i_nei:
                        cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                        if cos_theta >= 1.0:
                            cos_theta = 0.9999
                        theta = np.arccos(cos_theta)
                        phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                        if theta > np.pi/2.0:
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                        else:
                            beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            edges = np.delete(edges, 0, 0)
            edges[:, 2] *= lambda_c*round_factor
            edges = edges.astype(np.int32)

            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            # output refined result
            mesh3 = mesh_d.clone()
            mesh3.celldata['Label']=refine_labels
            return mesh3
            #vedo.write(mesh3, os.path.join(output_path, '{}_binary.vtp'.format(i_sample[:-4])))

            print('Sample filename: {} completed'.format(i_sample))







    def get_cshape(self,mesh,file_num):
        print("extracting c")



        label= mesh.celldata['Label']
        faces=mesh.faces()
        pt=mesh.points()
        ind = np.where(label==1)
        index=ind[0].tolist()
        new_face=[]
        for i in index:
            new_face.append(faces[i])

        mesh_n = Mesh([pt, new_face])
        #vedo.write(mesh_n, os.path.join('./src/predict/extracted_c', '{}.vtp'.format(os.path.splitext(file_num)[0])))
        vedo.write(mesh_n, os.path.join('./src/predict/extracted_c', file_num))























    
    
    def predict_labels(self, model_name, ip_path, op_path, file_num):

    
        gpu_id = 0
        #torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)
        upsampling_method = 'KNN'

        model_path = './src/models'
        model_name =model_name

        mesh_path = ip_path # need to modify
        #sample_filenames = ['{}.stl'.format(i) for i in range(2,len(os.listdir(ip_path)) + 1)] # need to modify
        #i_sample='{}.stl'.format(file_num)
        i_sample=file_num
        print(i_sample)
        

        output_path = op_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        num_classes = 17
        num_channels = 15

        # set model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

        # load trained model
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        model = model.to(device, dtype=torch.float)

        #cudnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


        # Predicting
        model.eval()
        with torch.no_grad():
            #for i_sample in sample_filenames:

            start_time = time.time()
            # create tmp folder
            tmp_path = './.tmp/'
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            print('Predicting Sample filename: {}'.format(i_sample))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, i_sample))

            # pre-processing: downsampling
            print('\tDownsampling...')
            target_num = 10000
            ratio = target_num/mesh.NCells() # calculate ratio
            mesh_d = mesh.clone()
            mesh_d.decimate(fraction=ratio)
            predicted_labels_d = np.zeros([mesh_d.NCells(), 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            cells = np.zeros([mesh_d.NCells(), 9], dtype='float32')
            for i in range(len(cells)):
                cells[i][0], cells[i][1], cells[i][2] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(0)) # don't need to copy
                cells[i][3], cells[i][4], cells[i][5] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(1)) # don't need to copy
                cells[i][6], cells[i][7], cells[i][8] = mesh_d.polydata().GetPoint(mesh_d.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            original_cells_d = cells.copy()

            mean_cell_centers = mesh_d.center_of_mass()
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh_d.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2)
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)

            mesh_normals[:, 0]= np.divide(mesh_normals[:, 0], mesh_normal_length[:], out=np.zeros_like(mesh_normals[:, 0]), where=mesh_normal_length[:]!=0)
            mesh_normals[:, 1]= np.divide(mesh_normals[:, 1], mesh_normal_length[:], out=np.zeros_like(mesh_normals[:, 1]), where=mesh_normal_length[:]!=0)
            mesh_normals[:, 2]= np.divide(mesh_normals[:, 2], mesh_normal_length[:], out=np.zeros_like(mesh_normals[:, 2]), where=mesh_normal_length[:]!=0)
            #mesh_normals[:, 0] /= mesh_normal_length[:]
            #mesh_normals[:, 1] /= mesh_normal_length[:]
            #mesh_normals[:, 2] /= mesh_normal_length[:]
            mesh_d.celldata['Normal']=mesh_normals

            # preprae input
            points = mesh_d.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            barycenters = mesh_d.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            #mesh2 = mesh_d.clone()
            #mesh2.celldata['Label']=predicted_labels_d
            #vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            # refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

            # unaries
            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            # parawise
            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            #edges
            normals = mesh_d.celldata['Normal'].copy() # need to copy, they use the same memory address
            cells = original_cells_d.copy()
            barycenters = mesh_d.cellCenters() # don't need to copy
            cell_ids = np.asarray(mesh_d.faces())

            lambda_c = 30
            edges = np.empty([1, 3], order='C')
            for i_node in range(cells.shape[0]):
                # Find neighbors
                nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
                nei_id = np.where(nei==2)
                for i_nei in nei_id[0][:]:
                    if i_node < i_nei:
                        cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                        if cos_theta >= 1.0:
                            cos_theta = 0.9999
                        theta = np.arccos(cos_theta)
                        phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                        if theta > np.pi/2.0:
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                        else:
                            beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            edges = np.delete(edges, 0, 0)
            edges[:, 2] *= lambda_c*round_factor
            edges = edges.astype(np.int32)

            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            # output refined result
            mesh3 = mesh_d.clone()
            mesh3.celldata['Label']=refine_labels
            vedo.write(mesh3, os.path.join(output_path, '{}_d_predicted_refined.vtp'.format(i_sample[:-4])))




        #    # upsampling
            # print('\tUpsampling...')
            # if mesh.NCells() > 100000:
            #     target_num = mesh.NCells() # set max number of cells
            #     ratio = target_num/mesh.NCells() # calculate ratio
            #     mesh.decimate(fraction=ratio)
            #     print('Original contains too many cells, simpify to {} cells'.format(mesh.NCells()))

            # # get fine_cells
            # cells = np.zeros([mesh.NCells(), 9], dtype='float32')
            # for i in range(len(cells)):
            #     cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
            #     cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
            #     cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy

            # fine_cells = cells

            # barycenters = mesh3.cellCenters() # don't need to copy
            # fine_barycenters = mesh.cellCenters() # don't need to copy

            # if upsampling_method == 'SVM':
            #     #clf = SVC(kernel='rbf', gamma='auto', probability=True, gpu_id=gpu_id)
            #     clf = SVC(kernel='rbf', gamma='auto', gpu_id=gpu_id)
            #     # train SVM
            #     #clf.fit(mesh2.cells, np.ravel(refine_labels))
            #     #fine_labels = clf.predict(fine_cells)

            #     clf.fit(barycenters, np.ravel(refine_labels))
            #     fine_labels = clf.predict(fine_barycenters)
            #     fine_labels = fine_labels.reshape(-1, 1)
            # elif upsampling_method == 'KNN':
            #     neigh = KNeighborsClassifier(n_neighbors=3)
            #     # train KNN
            #     #neigh.fit(mesh2.cells, np.ravel(refine_labels))
            #     #fine_labels = neigh.predict(fine_cells)

            #     neigh.fit(barycenters, np.ravel(refine_labels))
            #     fine_labels = neigh.predict(fine_barycenters)
            #     fine_labels = fine_labels.reshape(-1, 1)

            # mesh.celldata['Label']=fine_labels
            # vedo.write(mesh, os.path.join(output_path, '{}_predicted_refined.vtp'.format(i_sample[:-4])))

            #remove tmp folder
            shutil.rmtree(tmp_path)

            end_time = time.time()
            print('Sample filename: {} completed'.format(i_sample))
            print('\tcomputing time: {0:.2f} sec'.format(end_time-start_time))
            


                



# if __name__=="__main__": 

#     model_name ='MeshSegNet_17_classes_396_best.tar'
#     ip_path = './src/predict/input/itero_lower_set2'
#     op_path = './src/predict/output/itero_lower_set2_output'
#     exctract_c= False
#     p=PredictPipeline()



    

#     #for file_num in range(2, (len(os.listdir(ip_path)) + 1)): #try to change it to file name instead of a number
#     # 
#     for file_num in os.listdir(ip_path):
#         if exctract_c== True:
#             binary_mesh= p.predict_c(ip_path,file_num)
#             cshape=p.get_cshape(binary_mesh,file_num)
#             new_ip_path='./src/predict/extracted_c'
#             p.predict_labels(model_name, new_ip_path, op_path, file_num)
            

#         else:
#             p.predict_labels(model_name, ip_path, op_path, file_num)

