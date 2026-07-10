import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os 
import numpy as np
import random
import piq
import tqdm
import csv
import argparse

from variable_luminance import rgb_to_ycbcr
from Dataset import Compare_flickr1024, Compare_kitti2015, Compare_middlebury2006, StereoColorTransfer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id',     type=int, default = -1)
args = parser.parse_args()

if(args.dataset_id == -1):
    dataset_id = 1
else:
    dataset_id = args.dataset_id

dataset_id = 1

dataset_list = ['flickr1024','kitti2015', 'stereoColorTransferDataset', 'middlebury2006']
dataset = dataset_list[dataset_id]



model = ['ACE', 'RACE', 'Meur', 'SPACE', 'Proposed'] # ACE, RACE, Meur, SPACE, Proposed
model_path = ['exp0', 'exp0', 'exp0', 'exp0', 'exp0']
file_name = f'iqa_{dataset}.csv'
# file_name = f'iqa_ablation.csv'
class IQA:
    def __init__(self, resize=None, save=True):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ]) if resize is None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([480, 480])
        ])

        if dataset == 'flickr1024':
            self.dataset_path_gt = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
        elif dataset == 'kitti2015':
            self.dataset_path_gt = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'
            # self.dataset_path_gt2 = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'
        elif dataset == 'stereoColorTransferDataset':
            self.dataset_path_gt = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/stereoColorTransferDataset/Test'
        elif dataset == 'middlebury2006':
            self.dataset_path_gt = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        
        # print("GT Path:", self.dataset_path_gt)
        # print("Files in GT:", os.listdir(self.dataset_path_gt)[:5])
        
    def decompose_imgs(self, imgs, luminance_const=[0.206, 0.339, 0.454]):
        _ycbcr = rgb_to_ycbcr(imgs, luminance_const=luminance_const)
        _y = _ycbcr[:, :1]
        _cbcr = _ycbcr[:, 1:]
        return _y, _cbcr

    def EME(self, img, kernel_size=11, padding=5):
        _max_pool = F.max_pool2d(img, kernel_size=kernel_size, padding=padding, stride=kernel_size)
        _min_pool = F.max_pool2d(-img, kernel_size=kernel_size, padding=padding, stride=kernel_size) * -1
        _eme = 20 * (torch.log10((_max_pool + 1e-2) / (_min_pool + 1e-2)))
        return _eme.mean()

    def write_csv(self, row):
        with open(file_name, mode='a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def iqa(self): 
        with open(file_name, mode='a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset])

        for i in tqdm.tqdm(range(1, 10)):
            r = round(i * 0.1, 1)
            model_list = [f'{m}/{p}' for m, p in zip(model, model_path)]
            model_list.insert(0, f'R={r}')
            self.write_csv(model_list)

            print(f'\nR={r}')       
            ssim_list, vsi_list, eme_list = [], [], []
            vif_list, psnr_list = [], []

            for m in range(len(model)):
                print(model[m], model_path[m])
                result_path = f'C:/Users/lps3090/Desktop/Testing_StereoOLED/{model[m]}/{model_path[m]}/inference_{dataset}/R={r}'

                if dataset == 'flickr1024':
                    dataset_compare = Compare_flickr1024(root1=[self.dataset_path_gt], root2=[result_path], transform=self.transform)
                elif dataset == 'kitti2015':
                    dataset_compare = Compare_flickr1024(root1=[self.dataset_path_gt], root2=[result_path], transform=self.transform)
                    # dataset_compare = Compare_kitti2015(root1=[self.dataset_path_gt1, self.dataset_path_gt2], root2=[result_path], transform=self.transform)
                elif dataset == 'middlebury2006':
                    dataset_compare = Compare_middlebury2006(root1=[self.dataset_path_gt], root2=[result_path], transform=self.transform)
                elif dataset == 'stereoColorTransferDataset':
                    dataset_compare = StereoColorTransfer(root1=[self.dataset_path_gt], root2=[result_path], transform=self.transform)
                else:
                    print(f"⚠️ Unsupported dataset: {dataset}")
                    continue
                # print("Result Path:", result_path)
                # print("Files in Result:", os.listdir(result_path)[:5])

                
                print(f"Loaded {len(dataset_compare)} image pairs for {model[m]}")
                if len(dataset_compare) == 0:
                    print("⚠️ No image pairs loaded. Skipping.")
                    ssim_list.append('NaN')
                    vsi_list.append('NaN')
                    eme_list.append('NaN')
                    vif_list.append('NaN')
                    psnr_list.append('NaN')
                    continue

                test_loader = DataLoader(dataset_compare, batch_size=1, shuffle=False, num_workers=0)

                ssim_sum = vsi_sum = eme_sum = vif_sum = psnr_sum = 0
                count = 0

                for _, (input1, input2, filenameL, filenameR) in enumerate(test_loader):
                    gt_L, gt_R = input1[0], input1[1]
                    pred_L, pred_R = input2[0], input2[1]

                    ssim = (piq.ssim(gt_L, pred_L) + piq.ssim(gt_R, pred_R)) / 2
                    vsi = (piq.vsi(gt_L, pred_L) + piq.vsi(gt_R, pred_R)) / 2
                    eme = (self.EME(pred_L) + self.EME(pred_R)) / 2

                    ssim_sum += ssim
                    vsi_sum += vsi
                    eme_sum += eme

                    count += 1

                if count == 0:
                    print("⚠️ No images processed for this model.")
                    ssim_list.append('NaN')
                    vsi_list.append('NaN')
                    eme_list.append('NaN')
                    vif_list.append('NaN')
                    psnr_list.append('NaN')
                else:
                    ssim_list.append(float(ssim_sum / count))
                    vsi_list.append(float(vsi_sum / count))
                    eme_list.append(float(eme_sum / count))
                    vif_list.append(float(vif_sum / count))
                    psnr_list.append(float(psnr_sum / count))

                    print(f'ssimAvg = {ssim_sum/count:.4f}')
                    print(f'vsiAvg  = {vsi_sum/count:.4f}')
                    print(f'emeAvg  = {eme_sum/count:.4f}')

            ssim_list.insert(0, 'ssim')
            vsi_list.insert(0, 'vsi')
            eme_list.insert(0, 'eme')

            self.write_csv(ssim_list)
            self.write_csv(vsi_list)
            self.write_csv(eme_list)
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
iqa = IQA()
iqa.iqa()
