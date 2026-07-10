import kornia.image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import kornia
import os
import cv2
import tqdm

from models.space_net import SPACE
from utils import decompose_imgs, compose_imgs

from Dataset import testDataset, kittiTestDataset, middleburyTestDataset, stereoColorTransferDataset
 
model_path = 'exp0'

class Inference:
    def __init__(self, resize=None, save=True):
        pc = 1
        dataset_list = ['stereoColorTransferDataset','kitti2015', 'flickr1024','middlebury2006']
        dataset = dataset_list[1]

        num_workers = 0
        if(resize is None):
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([480,480])])
            
        if(pc == 0):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/stereoColorTransferDataset/Test'
                test_dataset = middleburyTestDataset(root=[dataset_test_path],transform=transform)

            elif(dataset == dataset_list[3]):   
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = stereoColorTransferDataset(root=[dataset_test_path],transform=transform)

        elif(pc == 1):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/stereoColorTransferDataset/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # test_dataset = middleburyTestDataset(root=[dataset_test_path],transform=transform)
                
        
            elif(dataset == dataset_list[1]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # dataset_test_path1 = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'
                # test_dataset = kittiTestDataset(root=[dataset_test_path0,dataset_test_path1],transform=transform)
            
            elif(dataset == dataset_list[2]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
            


            elif(dataset == dataset_list[3]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = stereoColorTransferDataset(root=[dataset_test_path],transform=transform)

        self.output_path = os.path.join(model_path,'inference_' + str(dataset))
        os.makedirs(self.output_path,exist_ok=True)

        torch.cuda.get_device_name(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = model_path
        self.model = SPACE(use_center_bias=True, use_gfcorrection=True, use_len=True)
        self.model.load_state_dict(torch.load("./pretrained_weights/salicon_space-cbgflen_sal.pth"))

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        
    def inference(self):        
        self.model.eval()

        problem_list = []
        with torch.no_grad():
            for batch_idx, (inputL, inputR, filenameL, filenameR) in tqdm.tqdm(enumerate(self.test_loader)):
                print('')
                print(filenameL)
                print(filenameR)
                R = 0.1
                R_in = R
                count = 0
                finish = False
                while(finish != True):
                    inputL_y, inputL_cbcr = decompose_imgs(inputL)
                    inputR_y, inputR_cbcr = decompose_imgs(inputR)
                    # Model inference
                    with torch.no_grad():
                
                        outL_y = self.model(inputL_y, R=R_in)
                        outR_y = self.model(inputR_y, R=R_in)

                    outL_rgb = torch.clamp(compose_imgs(outL_y, inputL_cbcr), 0., 1.)
                    outR_rgb = torch.clamp(compose_imgs(outR_y, inputR_cbcr), 0., 1.)

                    outL_rgb = outL_rgb.squeeze().permute(1,2,0)
                    outR_rgb = outR_rgb.squeeze().permute(1,2,0)

                    outL_rgb = outL_rgb.cpu().numpy()
                    outR_rgb = outR_rgb.cpu().numpy()

                    outL_rgb = np.uint8(outL_rgb*255)
                    outR_rgb = np.uint8(outR_rgb*255)
                    
                    outL_rgb = cv2.cvtColor(outL_rgb, cv2.COLOR_RGB2BGR)
                    outR_rgb = cv2.cvtColor(outR_rgb, cv2.COLOR_RGB2BGR)

                    img = np.concatenate([outL_rgb,outR_rgb],axis=1)
                    
                    cv2.imshow('0',img)
                    cv2.waitKey(1)

                    # R
                    p_in  = torch.sum(torch.pow(inputL_y + inputR_y, 2.2))
                    p_out = torch.sum(torch.pow(outL_y   + outR_y,   2.2))
                    R_actual = 1 - (p_out / p_in)

                    print(f'R = {R:.4f}, R_in = {R_in:.4f}, R_actual = {R_actual:.4f}')

                    if(count > 100):
                        problem_list.append([filenameL,filenameR,f'R = {R:.4f}, R_in = {R_in:.4f}, R_actual = {R_actual:.4f}'])

                        k = np.power((1 - R), (1/2.2))
                        outL_rgb = outL_rgb * k
                        outR_rgb = outR_rgb * k

                        count = 0
                        print(f'problem, R = {R:.4f}')
                        
                        save_path = os.path.join(self.output_path,'R='+str(R))
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path,filenameL[0]), outL_rgb)
                        cv2.imwrite(os.path.join(save_path,filenameR[0]), outR_rgb)

                        if(R > 0.85):
                            finish = True
                        R_in = R_in * ((R + 0.1) / R)
                        R = round(R + 0.1, 1)

                    count += 1
                    if(abs(R - R_actual) < 0.005):
                        count = 0
                        print(f'R       = {R:.4f}')
                        print(f'R_in    = {R_in:.4f}')
                        print(f'R_actual= {R_actual:.4f}\n')
                        save_path = os.path.join(self.output_path,'R='+str(R))
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path,filenameL[0]), outL_rgb)
                        cv2.imwrite(os.path.join(save_path,filenameR[0]), outR_rgb)

                        if(R_actual > 0.85):
                            finish = True
                        R_in = R_in * ((R + 0.1) / R)
                        R = round(R + 0.1, 1)
                    else:
                        R_in = R_in + float(R-R_actual) * (1 / torch.log10(torch.tensor(count)*10)+ 1e-5)
        for i in range(len(problem_list)):
            print(problem_list[i])

import random

def set_seed(seed=40, loader=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass
set_seed()

Inference = Inference()
Inference.inference()
