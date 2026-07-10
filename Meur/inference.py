import kornia.image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import kornia
import os
import cv2
import tqdm
import argparse
import random

from exp0.Model import Meur
from Dataset import testDataset, kittiTestDataset, middleburyTestDataset, Icvs2017

model_path = 'exp0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id',     type=int, default = -1)
args = parser.parse_args()

if(args.dataset_id != -1):
    dataset_id = 1
else:
    dataset_id = args.dataset_id

dataset_id = 1

dataset_list = ['flickr1024','kitti2015', 'stereoColorTransferDataset', 'middlebury2006']
dataset = dataset_list[dataset_id]

class Inference:
    def __init__(self, resize=None, save=True):
        pc = 1
        
        num_workers = 0
        if(resize is None):
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([480,480])])
            
        if(pc == 0):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)

            elif(dataset == dataset_list[3]):   
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = Icvs2017(root=[dataset_test_path],transform=transform)
    
        elif(pc == 1):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)

            elif(dataset == dataset_list[1]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # dataset_test_path1 = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'
                # test_dataset = kittiTestDataset(root=[dataset_test_path],transform=transform)

            elif(dataset == dataset_list[2]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/stereoColorTransferDataset/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # test_dataset = middleburyTestDataset(root=[dataset_test_path],transform=transform)

            elif(dataset == dataset_list[3]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = Icvs2017(root=[dataset_test_path],transform=transform)

        self.output_path = os.path.join(model_path,'inference_'+str(dataset))
        os.makedirs(self.output_path,exist_ok=True)

        torch.cuda.get_device_name(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = model_path
        self.model = Meur().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path,'best.pth')))

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        
    def inference(self):        
        self.model.eval()

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
                    # Left input
                    inputL_rgb = inputL.to(self.device)
                    inputL_ycbcr = kornia.color.rgb_to_ycbcr(inputL_rgb)
                    inputL_y = inputL_ycbcr[:,0:1,:,:]
                    inputL_cb = inputL_ycbcr[:,1:2,:,:]
                    inputL_cr = inputL_ycbcr[:,2:3,:,:]
                    # Right input
                    inputR_rgb = inputR.to(self.device)
                    inputR_ycbcr = kornia.color.rgb_to_ycbcr(inputR_rgb)
                    inputR_y = inputR_ycbcr[:,0:1,:,:]
                    inputR_cb = inputR_ycbcr[:,1:2,:,:]
                    inputR_cr = inputR_ycbcr[:,2:3,:,:]
                    # Model inference
                    outL_y = self.model(inputL_y,R_in)
                    outR_y = self.model(inputR_y,R_in)

                    outL_y = torch.where(torch.isnan(outL_y), 0, outL_y)
                    outR_y = torch.where(torch.isnan(outR_y), 0, outR_y)

                    outL_ycbcr = torch.cat((outL_y,inputL_cb,inputL_cr),1)
                    outR_ycbcr = torch.cat((outR_y,inputR_cb,inputR_cr),1)

                    outL_rgb = torch.clamp(kornia.color.ycbcr_to_rgb(outL_ycbcr),0,1)
                    outR_rgb = torch.clamp(kornia.color.ycbcr_to_rgb(outR_ycbcr),0,1)

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
                    p_in  = torch.mean(torch.pow(inputL_y + inputR_y, 2.2)) 
                    p_out = torch.mean(torch.pow(outL_y   + outR_y,   2.2)) 
                    R_actual = 1 - (p_out / p_in)

                    print(f'R = {R:.4f}, R_actual = {R_actual:.4f}, R_in = {R_in:.4f}')

                    if(abs(R - R_actual) < 0.005):
                        count = 0
                        print(f'R = {R:.4f}')
                        save_path = os.path.join(self.output_path,'R='+str(R))
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path,filenameL[0]), outL_rgb)
                        cv2.imwrite(os.path.join(save_path,filenameR[0]), outR_rgb)

                        if(R_actual > 0.85):
                            finish = True
                        R_in = R_in * ((R + 0.1) / R)
                        R = round(R + 0.1, 1)
                    else:
                        count += 1
                        R_in = R_in + float(R-R_actual) * 1

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
