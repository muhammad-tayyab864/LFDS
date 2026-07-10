import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import logging
import shutil
import tqdm
import yaml
import argparse

from Model import Proposed
from Model_new import MODEL
from Loss import ssim_loss, r_loss, contrast_loss, consistency_loss, cross_loss, TV_loss
from Dataset_aug import trainDataset
from variable_luminance import rgb_to_ycbcr, ycbcr_to_rgb


def set_seed(seed=40, loader=None):
    random.seed(seed)
    # np.random.seed(seed)
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

def decompose_imgs(imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    RGB to YCbCr, returns luminance and chrominance.
    """
    _ycbcr = rgb_to_ycbcr(imgs, luminance_const=luminance_const)
    _y = _ycbcr[:,:1]
    _cbcr = _ycbcr[:,1:]
    return _y, _cbcr

def compose_imgs(y_imgs, cbcr_imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    YCbCr to RGB, returns RGB.
    """
    _ycbcr = torch.cat((y_imgs, cbcr_imgs), dim=1)
    rgb_imgs = ycbcr_to_rgb(_ycbcr, luminance_const=luminance_const)
    return rgb_imgs


class Train:
    def __init__(self, config):
        # Laptop:  pc=0
        # Desktop: pc=1
        pc = 1
        if(pc == 0):
             dataset_train_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Train'
             dataset_val_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Validation'
        elif(pc == 1):
            dataset_train_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Train'
            dataset_val_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Validation'

        # Training param
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.img_shape = config['img_shape']
        # Model param
        self.hidden_size = config['hidden_size']            
        self.sampling_factor = config['sampling_factor']
        self.blending_factor = config['blending_factor']
        self.speedup = config['speedup']
        # Loss param
        self.pssim = config['pssim']
        self.pr = config['pr']
        self.pcross = config['pcross']
        self.pcontrast = config['pcontrast']
        self.pconsistency = config['pconsistency']
        self.ptv = config['ptv']

        self.num_workers = 0
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([self.img_shape,self.img_shape])])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {torch.cuda.get_device_name(0)}')
        # Set seed
        set_seed()
        # Data loader
        self.train_loader, self.eval_loader = self.data_loader(dataset_train_path, dataset_val_path, transform)
        # Output folder
        self.out_file = self.save_folder(config)
        # Logger
        self.logging = self.logger()
        
    def train(self, train_loader, eval_loader):
        model = Proposed(hidden_size=self.hidden_size, sampling_factor=self.sampling_factor, blending_factor=self.blending_factor, speedup=self.speedup).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate) # AdamW
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999)) # Adam
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)//self.batch_size)*self.num_epochs)
        

        rgb = 0

        self.logging.info(model)
        train_list = []
        val_list = []
        min_loss = 10000
        min_loss_epoch = 0

        r_list = torch.linspace(0.1,0.9,90).cuda()

        start_time = time.time()
        self.logging.info('Start Training')
        print(f'Start Training')
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            print('')
            print(self.out_file)
            ### Train
            model.train()

            ssim_loss_train = 0
            r_loss_train = 0
            contrast_loss_train = 0
            cross_loss_train = 0
            consistency_loss_train = 0
            TV_loss_train = 0
            count_train = 0
            valid_train = 0
            for batch_idx, (dataL, dataR) in enumerate(train_loader):
                dataL = dataL.to(self.device)
                dataR = dataR.to(self.device)
                dataL = Variable(dataL)
                dataR = Variable(dataR)

                dataL_y, cbcrL = decompose_imgs(dataL)
                dataR_y, cbcrR = decompose_imgs(dataR)

                optimizer.zero_grad()
                r = random.choice(r_list)
                
                if(rgb == 0):
                    outL_y, outR_y, M_right_to_left, M_left_to_right, V_left, V_right = model(dataL_y, dataR_y, r)
                    outL = torch.clamp(compose_imgs(outL_y, cbcrL), 0, 1)
                    outR = torch.clamp(compose_imgs(outR_y, cbcrR), 0, 1)
                else:
                    outL, outR, M_right_to_left, M_left_to_right, V_left, V_right = model(dataL, dataR, r)
                    outL_y, _ = decompose_imgs(outL)
                    outR_y, _ = decompose_imgs(outR)

                

                valid_train += (V_left.mean() + V_right.mean()) / 2

                # Train Loss
                _ssimloss = self.pssim*(ssim_loss(outL_y, dataL_y) + ssim_loss(outR_y, dataR_y))
                _crossloss = self.pcross*(cross_loss(dataL_y,dataR_y, M_right_to_left,M_left_to_right, V_left,V_right, sampling_factor=self.sampling_factor))
                _consistencyloss = self.pconsistency*(consistency_loss(outL_y,outR_y, M_right_to_left,M_left_to_right, V_left,V_right, sampling_factor=self.sampling_factor))
                _rloss =  self.pr*(r_loss(outL_y, dataL_y, r) + r_loss(outR_y, dataR_y, r))
                _TVloss =  self.ptv*(TV_loss(outL_y, dataL_y) + TV_loss(outR_y, dataR_y))
                _contrastloss =  self.pcontrast*(contrast_loss(outL_y, dataL_y, M_right_to_left, V_left, sampling_factor=self.sampling_factor) + contrast_loss(outR_y, dataR_y, M_left_to_right, V_right, sampling_factor=self.sampling_factor))

                loss = _ssimloss + _crossloss + _TVloss + _rloss + _contrastloss + _consistencyloss
                
                # Optimize
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Log
                ssim_loss_train  += _ssimloss.item()
                cross_loss_train += _crossloss.item()
                consistency_loss_train += _consistencyloss.item()
                TV_loss_train += _TVloss.item()
                r_loss_train  += _rloss.item()
                contrast_loss_train += _contrastloss.item()

                count_train += 1
                avg_loss_train = (ssim_loss_train + r_loss_train + contrast_loss_train + cross_loss_train + TV_loss_train + consistency_loss_train) / count_train
                
                if batch_idx % (len(train_loader)//5) == 0:
                    print(f'Train Epoch [{epoch}/{self.num_epochs}], Step [{batch_idx}/{len(train_loader)}], Avg_Loss: {avg_loss_train:.4f}, Valid: {(valid_train/count_train):.4f}')

            print(f'Train Epoch [{epoch}/{self.num_epochs}], Avg_loss: {avg_loss_train:.4f}, Valid: {(valid_train/count_train):.4f}')
            self.logging.info(f'Train Epoch [{epoch}/{self.num_epochs}], Avg_loss: {avg_loss_train:.4f}, Valid: {(valid_train/count_train):.4f}')
            ### Eval
            with torch.no_grad():
                model.eval()
                ssim_loss_val = 0
                r_loss_val = 0
                contrast_loss_val = 0
                consistency_loss_val = 0
                cross_loss_val = 0
                TV_loss_val = 0
                count_val = 0
                valid_val = 0

                for batch_idx, (dataL, dataR) in enumerate(eval_loader):
                    dataL = dataL.to(self.device)
                    dataR = dataR.to(self.device)
                    dataL = Variable(dataL)
                    dataR = Variable(dataR)

                    dataL_y, cbcrL = decompose_imgs(dataL)
                    dataR_y, cbcrR = decompose_imgs(dataR)

                    r = random.choice(r_list)

                    if(rgb == 0):
                        outL_y, outR_y, M_right_to_left, M_left_to_right, V_left, V_right = model(dataL_y, dataR_y, r)
                        outL = torch.clamp(compose_imgs(outL_y, cbcrL), 0, 1)
                        outR = torch.clamp(compose_imgs(outR_y, cbcrR), 0, 1)
                    else:
                        outL, outR, M_right_to_left, M_left_to_right, V_left, V_right = model(dataL, dataR, r)
                        outL_y, _ = decompose_imgs(outL)
                        outR_y, _ = decompose_imgs(outR)


                    valid_val += (V_left.mean() + V_right.mean()) / 2

                    # Eval Loss
                    _ssimloss = self.pssim*(ssim_loss(outL_y, dataL_y) + ssim_loss(outR_y, dataR_y))
                    _crossloss = self.pcross*(cross_loss(dataL_y,dataR_y, M_right_to_left,M_left_to_right, V_left,V_right, sampling_factor=self.sampling_factor))
                    _consistencyloss = self.pconsistency*(consistency_loss(outL_y,outR_y, M_right_to_left,M_left_to_right, V_left,V_right, sampling_factor=self.sampling_factor))
                    _rloss =  self.pr*(r_loss(outL_y, dataL_y, r) + r_loss(outR_y, dataR_y, r))
                    _TVloss =  self.ptv*(TV_loss(outL_y, dataL_y) + TV_loss(outR_y, dataR_y))
                    _contrastloss =  self.pcontrast*(contrast_loss(outL_y, dataL_y, M_right_to_left, V_left, sampling_factor=self.sampling_factor) + contrast_loss(outR_y, dataR_y, M_left_to_right, V_right, sampling_factor=self.sampling_factor))

                    loss = _ssimloss + _crossloss + _TVloss + _rloss + _contrastloss + _consistencyloss
                    
                    # Log
                    ssim_loss_val += _ssimloss.item()
                    cross_loss_val += _crossloss.item()
                    consistency_loss_val += _consistencyloss.item()
                    TV_loss_val += _TVloss.item()
                    r_loss_val += _rloss.item()
                    contrast_loss_val += _contrastloss.item()
                    count_val += 1
                    avg_loss_val = (ssim_loss_val + r_loss_val + contrast_loss_val + cross_loss_val + TV_loss_val + consistency_loss_val) / count_val

                # Save and Log
                train_list.append([ssim_loss_train/count_train, r_loss_train/count_train, cross_loss_train/count_train, contrast_loss_train/count_train, TV_loss_train/count_train, consistency_loss_train/count_train])
                val_list.append(  [ssim_loss_val/count_val,     r_loss_val/count_val,     cross_loss_val/count_val,     contrast_loss_val/count_val,     TV_loss_val/count_val,     consistency_loss_val/count_val])

                torch.save(model.state_dict(), os.path.join(self.out_file,'last.pth'))
                if(avg_loss_val < min_loss):
                    min_loss_epoch = epoch
                    min_loss = avg_loss_val
                    torch.save(model.state_dict(), os.path.join(self.out_file,'best.pth'))
                print(f'Valid Eval :{(valid_val/count_val):.4f}')
                print(f'SSIM        Eval Loss:{ssim_loss_val/count_val:.4f}')
                print(f'R           Eval Loss:{r_loss_val/count_val:.4f}')
                print(f'contrast    Eval Loss:{contrast_loss_val/count_val:.4f}')
                print(f'consistency Eval Loss:{consistency_loss_val/count_val:.4f}')
                print(f'cross       Eval Loss:{cross_loss_val/count_val:.4f}')
                print(f'TV          Eval Loss:{TV_loss_val/count_val:.4f}')
                print(f'Avg         Eval Loss:{avg_loss_val:.4f}, Min Loss Epoch: {min_loss_epoch}, Min Loss: {min_loss:.4f}\n')

                self.logging.info(f'Valid Eval :{(valid_val/count_val):.4f}')
                self.logging.info(f'SSIM        Eval Loss:{ssim_loss_val/count_val:.4f}')
                self.logging.info(f'R           Eval Loss:{r_loss_val/count_val:.4f}')
                self.logging.info(f'contrast    Eval Loss:{contrast_loss_val/count_val:.4f}')
                self.logging.info(f'consistency Eval Loss:{consistency_loss_val/count_val:.4f}')
                self.logging.info(f'cross       Eval Loss:{cross_loss_val/count_val:.4f}')
                self.logging.info(f'TV          Eval Loss:{TV_loss_val/count_val:.4f}')
                self.logging.info(f'Avg         Eval Loss:{avg_loss_val:.4f}, Min Loss Epoch: {min_loss_epoch}, Min Loss: {min_loss:.4f}\n')


        end_time=time.time()
        print(f'End Training')
        print(f'Training for {end_time-start_time}')
        self.logging.info('End Training')
        print(f'Results save in {self.out_file}')

        # Plot Results
        t = np.linspace(1,len(train_list),len(train_list))

        ssim_train = [train_list[i][0] for i in range(len(train_list))]
        ssim_val = [val_list[i][0] for i in range(len(val_list))]
        r_train = [train_list[i][1] for i in range(len(train_list))]
        r_val = [val_list[i][1] for i in range(len(val_list))]
        cross_train = [train_list[i][2] for i in range(len(train_list))]
        cross_val = [val_list[i][2] for i in range(len(val_list))]
        contrast_train = [train_list[i][3] for i in range(len(train_list))]
        contrast_val = [val_list[i][3] for i in range(len(val_list))]
        TV_train = [train_list[i][4] for i in range(len(train_list))]
        TV_val = [val_list[i][4] for i in range(len(val_list))]
        consistency_train = [train_list[i][5] for i in range(len(train_list))]
        consistency_val = [val_list[i][5] for i in range(len(val_list))]
        ## SSIM Loss
        plt.subplot(231)
        plt.plot(t,ssim_train,label='ssim train')
        plt.plot(t,ssim_val,label='ssim val')
        plt.title('SSIM Loss')
        plt.legend()
        ## R Loss
        plt.subplot(232)
        plt.plot(t,r_train,label='r train')
        plt.plot(t,r_val,label='r val')
        plt.title('R Loss')
        plt.legend()
        ## cross Loss
        plt.subplot(233)
        plt.plot(t,cross_train,label='cross train')
        plt.plot(t,cross_val,label='cross val')
        plt.title('cross Loss')
        plt.legend()
        ## contrast Loss
        plt.subplot(234)
        plt.plot(t,contrast_train,label='contrast train')
        plt.plot(t,contrast_val,label='contrast val')
        plt.title('contrast Loss')
        plt.legend()
        ## TV Loss
        plt.subplot(235)
        plt.plot(t,TV_train,label='TV train')
        plt.plot(t,TV_val,label='TV val')
        plt.title('TV Loss')
        plt.legend()
        ## consistency Loss
        plt.subplot(236)
        plt.plot(t,consistency_train,label='consistency train')
        plt.plot(t,consistency_val,label='consistency val')
        plt.title('consistency Loss')
        plt.legend()

        plt.savefig(os.path.join(self.out_file,'training_loss.png'))
    
    def start_train(self):
        train_loader = self.train_loader
        eval_loader = self.eval_loader

        self.train(train_loader,eval_loader)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)    

    def data_loader(self, dataset_train_path, dataset_val_path, transform):
        train_dataset = trainDataset(root=[dataset_train_path],transform=transform)
        eval_dataset = trainDataset(root=[dataset_val_path],transform=transform)

        print(f'Train dataset:\n \
              \t-Path: {dataset_train_path}\n \
              \t-Data: {len(train_dataset)}, {train_dataset[1][0].size()}')
        print(f'Eval  dataset:\n \
              \t-Path: {dataset_val_path}\n \
              \t-Data: {len(eval_dataset) }, {eval_dataset[1][0].size() }')
        g = torch.Generator()
        g.manual_seed(0)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, worker_init_fn=self.seed_worker, generator=g, pin_memory=True)
        eval_loader  = DataLoader(eval_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return train_loader, eval_loader
        
    def save_folder(self, config):
        out_file = 'exp'
        i = 0
        out_file_tmp = out_file + str(i)
        while(os.path.isdir(out_file_tmp) == True):
            i += 1
            out_file_tmp = out_file + str(i)
        out_file = out_file + str(i)
        os.makedirs(out_file, exist_ok=True)
        print(f'Results save in {out_file}')
        # Copy model and loss
        shutil.copy('Model.py', out_file)
        shutil.copy('Loss.py', out_file)
        shutil.copy('train.py', out_file)
        with open(os.path.join(out_file, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        return out_file
    
    def logger(self):
        logging.basicConfig(filename=os.path.join(self.out_file,"log.txt"), level=logging.DEBUG, format="%(asctime)s %(message)s")
        return logging

if __name__ == '__main__':

    os.chdir(r'C:/Users/lps3090/Desktop/StereoOLED/bbiPAM')

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size',     type=int, default = -1)
    parser.add_argument('--sampling_factor', type=int, default = -1)
    parser.add_argument('--blending_factor', type=int, default = -1)
    parser.add_argument('--speedup',         type=int, default = -1)

    parser.add_argument('--pcontrast',       type=int, default = -1)
    parser.add_argument('--pconsistency',    type=int, default = -1)
    parser.add_argument('--ptv',             type=int, default = -1)
    args = parser.parse_args()

    if(args.hidden_size != -1):
        config['hidden_size'] = args.hidden_size

    if(args.sampling_factor != -1):
        config['sampling_factor'] = args.sampling_factor

    if(args.blending_factor != -1):
        config['blending_factor'] = args.blending_factor

    if(args.speedup != -1):
        config['speedup'] = args.speedup
    
    if(args.pcontrast != -1):
        config['pcontrast'] = args.pcontrast

    if(args.pconsistency != -1):
        config['pconsistency'] = args.pconsistency

    if(args.ptv != -1):
        config['ptv'] = args.ptv

    train = Train(config)
    train.start_train() 
