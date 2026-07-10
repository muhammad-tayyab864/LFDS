import os
import cv2
import glob
import random
import numpy as np
from torch.utils.data.dataset import Dataset

# =====================================
#          == Augmentation ==
# =====================================
def augmentation(image_left, image_right):

    if random.random()<0.5:     #flip horizonly
        image_left = image_left[:, ::-1, :]
        image_right = image_right[:, ::-1, :]

    if random.random()<0.5:     #flip vertically
        image_left = image_left[::-1, :, :]
        image_right = image_right[::-1, :, :]

    if random.random()<0.5:     # rotation
        image_left = image_left.transpose(1, 0, 2)
        image_right = image_right.transpose(1, 0, 2)

    return np.ascontiguousarray(image_left), np.ascontiguousarray(image_right)

# =====================================
#     == Training Dataloader ==
# =====================================
class trainDataset(Dataset):
    def __init__(self,root,transform):
        self.transform = transform
        self.imgs_path = root
        self.data = []

        for i in range(len(self.imgs_path)):
            self.data_list_L = []
            self.data_list_R = []
            file_list_L = glob.glob(self.imgs_path[i] + "*/*L.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "*/*L.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*L.bmp",recursive=True)
            file_list_R = glob.glob(self.imgs_path[i] + "*/*R.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "*/*R.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*R.bmp",recursive=True)
            for j in range(len(file_list_L)):
                self.data_list_L.append(file_list_L[j])
            for j in range(len(file_list_R)):
                self.data_list_R.append(file_list_R[j])

            for j in range(len(self.data_list_L)):
                self.data.append([self.data_list_L[j],self.data_list_R[j]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        k=8
        imgL = cv2.resize(imgL,((w//k)*k,(h//k)*k))
        imgR = cv2.resize(imgR,((w//k)*k,(h//k)*k))

        imgL, imgR = augmentation(imgL, imgR)

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor

# =====================================
#   == Flicker Testing Dataloader ==
# =====================================
class testDataset(Dataset):
    def __init__(self,root,transform):
        self.transform = transform
        self.imgs_path = root
        self.data = []
        
        for i in range(len(self.imgs_path)):
            self.data_list_L = []
            self.data_list_R = []
            file_list_L = glob.glob(self.imgs_path[i] + "*/*L.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "*/*L.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*L.bmp",recursive=True)
            file_list_R = glob.glob(self.imgs_path[i] + "*/*R.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "*/*R.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*R.bmp",recursive=True)
            for j in range(len(file_list_L)):
                self.data_list_L.append(file_list_L[j])
            for j in range(len(file_list_R)):
                self.data_list_R.append(file_list_R[j])

            for j in range(len(self.data_list_L)):
                self.data.append([self.data_list_L[j],self.data_list_R[j]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        filenameL = os.path.split(imgL_path)[-1]
        filenameR = os.path.split(imgR_path)[-1]
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        k=8
        imgL = cv2.resize(imgL,((w//k)*k,(h//k)*k))
        imgR = cv2.resize(imgR,((w//k)*k,(h//k)*k))

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor, filenameL, filenameR

# =====================================
#    == KITTI Testing Dataloader ==
# =====================================
class kittiTestDataset(Dataset):
    def __init__(self,root,transform):
        self.transform = transform
        self.imgs_path = root
        self.data = []
        
        self.data_list_L = []
        self.data_list_R = []

        file_list_L = glob.glob(self.imgs_path[0] + "*/*.jpg",recursive=True) \
                    + glob.glob(self.imgs_path[0] + "*/*.png",recursive=True) \
                    + glob.glob(self.imgs_path[0] + "/**/*.bmp",recursive=True)
        
        file_list_R = glob.glob(self.imgs_path[1] + "*/*.jpg",recursive=True) \
                    + glob.glob(self.imgs_path[1] + "*/*.png",recursive=True) \
                    + glob.glob(self.imgs_path[1] + "/**/*.bmp",recursive=True)
        

        for i in range(len(file_list_L)):
            self.data.append([file_list_L[i], file_list_R[i]])
        for i in range(len(self.data)):
            print(self.data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        filenameL =  os.path.split(imgL_path)[-1][:-4] + '_' + os.path.split(os.path.split(imgL_path)[-2])[-1] + '.png'
        filenameR =  os.path.split(imgR_path)[-1][:-4] + '_' + os.path.split(os.path.split(imgR_path)[-2])[-1] + '.png'
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        k=8
        imgL = cv2.resize(imgL,((w//k)*k,(h//k)*k))
        imgR = cv2.resize(imgR,((w//k)*k,(h//k)*k))

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor, filenameL, filenameR

# ======================================
#  == middlebury Testing Dataloader ==
# ======================================
class middleburyTestDataset(Dataset):
    def __init__(self,root,transform):
        self.transform = transform
        self.imgs_path = root
        self.data = []
        
        for i in range(len(self.imgs_path)):
            self.data_list_L = []
            self.data_list_R = []
            file_list_L = glob.glob(self.imgs_path[i] + "/**/*view1.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*view1.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*view1.bmp",recursive=True)
            file_list_R = glob.glob(self.imgs_path[i] + "/**/*view5.jpg",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*view5.png",recursive=True) \
                        + glob.glob(self.imgs_path[i] + "/**/*view5.bmp",recursive=True)
            for j in range(len(file_list_L)):
                self.data_list_L.append(file_list_L[j])
            for j in range(len(file_list_R)):
                self.data_list_R.append(file_list_R[j])

            for j in range(len(self.data_list_L)):
                self.data.append([self.data_list_L[j],self.data_list_R[j]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        filenameL = os.path.split(os.path.split(os.path.split(os.path.split(imgL_path)[-2])[-2])[-2])[-1] + '_' + os.path.split(os.path.split(os.path.split(imgL_path)[-2])[-2])[-1] + '_' + os.path.split(os.path.split(imgL_path)[-2])[-1] + '_' + os.path.split(imgL_path)[-1]
        filenameR = os.path.split(os.path.split(os.path.split(os.path.split(imgR_path)[-2])[-2])[-2])[-1] + '_' + os.path.split(os.path.split(os.path.split(imgR_path)[-2])[-2])[-1] + '_' + os.path.split(os.path.split(imgR_path)[-2])[-1] + '_' + os.path.split(imgR_path)[-1]
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        k=8
        imgL = cv2.resize(imgL,((w//k)*k,(h//k)*k))
        imgR = cv2.resize(imgR,((w//k)*k,(h//k)*k))

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor, filenameL, filenameR

 
class Compare(Dataset):
    def __init__(self,root1,root2,transform):
        self.transform = transform
        self.imgs_path1 = root1
        self.data1 = []
        
        for i in range(len(self.imgs_path1)):
            self.data_list1_L = []
            self.data_list1_R = []
            file_list1_L = glob.glob(self.imgs_path1[i] + "*/*L.jpg",recursive=True) \
                        + glob.glob(self.imgs_path1[i] + "*/*L.png",recursive=True) \
                        + glob.glob(self.imgs_path1[i] + "/**/*L.bmp",recursive=True)
            file_list1_R = glob.glob(self.imgs_path1[i] + "*/*R.jpg",recursive=True) \
                        + glob.glob(self.imgs_path1[i] + "*/*R.png",recursive=True) \
                        + glob.glob(self.imgs_path1[i] + "/**/*R.bmp",recursive=True)
            for j in range(len(file_list1_L)):
                self.data_list1_L.append(file_list1_L[j])
            for j in range(len(file_list1_R)):
                self.data_list1_R.append(file_list1_R[j])

            for j in range(len(self.data_list1_L)):
                self.data1.append([self.data_list1_L[j],self.data_list1_R[j]])

        self.imgs_path2 = root2
        self.data2 = []
        self.data_list2_L = []
        self.data_list2_R = []

        for i in range(len(self.imgs_path2)):
            file_list2_L = glob.glob(self.imgs_path2[i] + "*/*L.jpg",recursive=True) \
                        + glob.glob(self.imgs_path2[i] + "*/*L.png",recursive=True) \
                        + glob.glob(self.imgs_path2[i] + "/**/*L.bmp",recursive=True)
            file_list2_R = glob.glob(self.imgs_path2[i] + "*/*R.jpg",recursive=True) \
                        + glob.glob(self.imgs_path2[i] + "*/*R.png",recursive=True) \
                        + glob.glob(self.imgs_path2[i] + "/**/*R.bmp",recursive=True)
            for j in range(len(file_list2_L)):
                self.data_list2_L.append(file_list2_L[j])
            for j in range(len(file_list2_R)):
                self.data_list2_R.append(file_list2_R[j])

            for j in range(len(self.data_list2_L)):
                self.data2.append([self.data_list2_L[j],self.data_list2_R[j]])

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        imgL_path1 = self.data1[idx][0]
        imgR_path1 = self.data1[idx][1]

        img1L = cv2.imread(imgL_path1)
        img1R = cv2.imread(imgR_path1)
        img1L = cv2.cvtColor(img1L,cv2.COLOR_BGR2RGB)
        img1R = cv2.cvtColor(img1R,cv2.COLOR_BGR2RGB)

        h,w = img1L.shape[:2]
        img1L = cv2.resize(img1L,((w//4)*4,(h//4)*4))
        img1R = cv2.resize(img1R,((w//4)*4,(h//4)*4))

        img1L_tensor = self.transform(img1L)
        img1R_tensor = self.transform(img1R)

        imgL_path2 = self.data2[idx][0]
        imgR_path2 = self.data2[idx][1]

        img2L = cv2.imread(imgL_path2)
        img2R = cv2.imread(imgR_path2)
        img2L = cv2.cvtColor(img2L,cv2.COLOR_BGR2RGB)
        img2R = cv2.cvtColor(img2R,cv2.COLOR_BGR2RGB)

        h,w = img2L.shape[:2]
        img2L = cv2.resize(img2L,((w//4)*4,(h//4)*4))
        img2R = cv2.resize(img2R,((w//4)*4,(h//4)*4))

        img2L_tensor = self.transform(img2L)
        img2R_tensor = self.transform(img2R)

        filenameL = os.path.split(imgL_path1)[-1] + '/' + os.path.split(imgL_path2)[-1]
        filenameR = os.path.split(imgR_path1)[-1] + '/' + os.path.split(imgR_path2)[-1]

        return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR

class Icvs2017(Dataset):
    def __init__(self,root,transform):
        self.transform = transform
        self.imgs_path = root
        self.data = []
        
        for i in range(len(self.imgs_path)):
            self.data_list_L = []
            self.data_list_R = []
            file_list_L = glob.glob(self.imgs_path[i] + "/left/*.jpg",  recursive=True) 
            
            file_list_R = glob.glob(self.imgs_path[i] + "/right/*.jpg", recursive=True) 

            for j in range(len(file_list_L)):
                self.data_list_L.append(file_list_L[j])
            for j in range(len(file_list_R)):
                self.data_list_R.append(file_list_R[j])
            
            for j in range(len(self.data_list_L)):
                self.data.append([self.data_list_L[j],self.data_list_R[j]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        imgL = cv2.resize(imgL,((w//4)*4,(h//4)*4))
        imgR = cv2.resize(imgR,((w//4)*4,(h//4)*4))

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor
    
class MPI_Sintel(Dataset):
    def __init__(self,rootL,rootR,transform):
        self.transform = transform
        self.imgs_pathL = rootL
        self.imgs_pathR = rootR
        self.data = []
        
        for i in range(len(self.imgs_pathL)):
            self.data_list_L = []
            self.data_list_R = []
            file_list_L = glob.glob(self.imgs_pathL[i] + "/*.png",  recursive=True) 
            
            file_list_R = glob.glob(self.imgs_pathR[i] + "/*.png", recursive=True) 

            for j in range(len(file_list_L)):
                self.data_list_L.append(file_list_L[j])
            for j in range(len(file_list_R)):
                self.data_list_R.append(file_list_R[j])
            
            for j in range(len(self.data_list_L)):
                self.data.append([self.data_list_L[j],self.data_list_R[j]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgL_path = self.data[idx][0]
        imgR_path = self.data[idx][1]
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

        h,w = imgL.shape[:2]
        imgL = cv2.resize(imgL,((w//4)*4,(h//4)*4))
        imgR = cv2.resize(imgR,((w//4)*4,(h//4)*4))

        imgL_tensor = self.transform(imgL)
        imgR_tensor = self.transform(imgR)

        return imgL_tensor, imgR_tensor

