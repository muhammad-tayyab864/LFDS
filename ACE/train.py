import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import kornia
import random
import time
import os
import logging
import shutil
import tqdm

from Model import ACE
from Loss import ssim_loss, r_loss, contrast_loss, TV_loss
from Dataset import trainDataset
from variable_luminance import rgb_to_ycbcr, ycbcr_to_rgb

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

class Train:
    def __init__(self,num_epochs = 300, batch_size = 1, num_workers = 0, img_shape = (450,450)):
        pc = 1
        if(pc == 0):
             dataset_train_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Train'
             dataset_val_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Validation'
        elif(pc == 1):
            dataset_train_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Train'
            dataset_val_path = 'C:/Users/lps3090/Desktop/StereoOLED/Datasets/stereoColorTransferDataset/Validation'

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([img_shape[0],img_shape[1]])])

        print(f'Using {torch.cuda.get_device_name(0)}')
        # set seed
        set_seed()
        # data loader
        self.train_loader, self.eval_loader = self.data_loader(dataset_train_path, dataset_val_path, transform, batch_size, num_workers)
        # output folder
        self.out_file = self.save_folder()
        # Logger
        self.logging = self.logger()
        
    def train(self,num_epochs,train_loader,eval_loader):
        ace = ACE().to(self.device)
        optimizer = optim.Adam(ace.parameters(), lr=ace.learning_rate, betas=(0.9, 0.999))
        
        self.logging.info(ace)

        train_list = []
        val_list = []
        min_loss = 10000
        min_loss_epoch = 0

        r_list = np.linspace(0.1,0.9,90)

        start_time = time.time()
        self.logging.info('Start Training')
        print(f'Start Training')
        for epoch in tqdm.tqdm(range(num_epochs)):
            print('')
            ### Train
            ace.train()

            ssim_loss_train = 0
            r_loss_train = 0
            contrast_loss_train = 0
            count_train = 0
            for batch_idx, (dataL, dataR) in enumerate(train_loader):
                dataL = Variable(dataL).to(self.device)
                dataR = Variable(dataR).to(self.device)

                dataL_y, cbcrL = decompose_imgs(dataL)
                dataR_y, cbcrR = decompose_imgs(dataR)

                optimizer.zero_grad()
                
                r = random.choice(r_list)

                outL_y = ace(dataL_y, r)
                outR_y = ace(dataR_y, r)
                
                outL = torch.clamp(compose_imgs(outL_y, cbcrL), 0, 1)
                outR = torch.clamp(compose_imgs(outR_y, cbcrR), 0, 1)

                # Train Loss
                _ssimloss = ssim_loss(outL, dataL) + ssim_loss(outR, dataR)
                _rloss =  r_loss(outL_y, dataL_y, r) + r_loss(outR_y, dataR_y, r)
                _contrastloss = contrast_loss(outL_y, dataL_y, r) + contrast_loss(outR_y, dataR_y, r)
                loss = (_ssimloss + _rloss + _contrastloss) / 2

                # Optimize
                loss.backward()
                optimizer.step()
                
                # Log
                ssim_loss_train += _ssimloss.item()
                r_loss_train += _rloss.item()
                contrast_loss_train += _contrastloss.item()
                count_train += 1
                avg_loss_train = (ssim_loss_train + r_loss_train + contrast_loss_train)/count_train
                
                if batch_idx % (len(train_loader)//5) == 0:
                    print(f'Train Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Avg_Loss: {avg_loss_train}')

            print(f'Train Epoch [{epoch}/{num_epochs}], Avg_loss: {avg_loss_train}')
            self.logging.info(f'Train Epoch [{epoch}/{num_epochs}], Avg_loss: {avg_loss_train}')
            ### Eval
            with torch.no_grad():
                ace.eval()
                ssim_loss_val = 0
                r_loss_val = 0
                contrast_loss_val = 0
                count_val = 0

                for batch_idx, (dataL, dataR) in enumerate(eval_loader):
                    dataL = Variable(dataL).to(self.device)
                    dataR = Variable(dataR).to(self.device)

                    dataL_y, cbcrL = decompose_imgs(dataL)
                    dataR_y, cbcrR = decompose_imgs(dataR)

                    optimizer.zero_grad()
                    
                    r = random.choice(r_list)

                    outL_y = ace(dataL_y, r)
                    outR_y = ace(dataR_y, r)
                    
                    outL = torch.clamp(compose_imgs(outL_y, cbcrL), 0, 1)
                    outR = torch.clamp(compose_imgs(outR_y, cbcrR), 0, 1)
                    # Eval Loss
                    _ssimloss = ssim_loss(outL, dataL) + ssim_loss(outR, dataR)
                    _rloss =  r_loss(outL_y, dataL_y, r) + r_loss(outR_y, dataR_y, r)
                    _contrastloss = contrast_loss(outL_y, dataL_y, r) + contrast_loss(outR_y, dataR_y, r)
                    loss = (_ssimloss + _rloss + _contrastloss) / 2
                    # Log
                    ssim_loss_val += _ssimloss.item()
                    r_loss_val += _rloss.item()
                    contrast_loss_val += _contrastloss.item()
                    count_val += 1
                    avg_loss_val = (ssim_loss_val + r_loss_val + contrast_loss_val)/count_val

                ### save and log
                train_list.append([ssim_loss_train/count_train, r_loss_train/count_train, contrast_loss_train/count_train])
                val_list.append([ssim_loss_val/count_val, r_loss_val/count_val, contrast_loss_val/count_val])

                torch.save(ace.state_dict(), os.path.join(self.out_file,'last.pth'))
                if(avg_loss_val < min_loss):
                    min_loss_epoch = epoch
                    min_loss = avg_loss_val
                    torch.save(ace.state_dict(), os.path.join(self.out_file,'best.pth'))
                print(f'SSIM     Eval Loss:{ssim_loss_val/count_val:.4f}')
                print(f'R        Eval Loss:{r_loss_val/count_val:.4f}')
                print(f'contrast Eval Loss:{contrast_loss_val/count_val:.4f}')
                print(f'Avg      Eval Loss:{avg_loss_val:.4f}, Min Loss Epoch: {min_loss_epoch}, Min Loss: {min_loss}\n')

                self.logging.info(f'SSIM     Eval Loss:{ssim_loss_val/count_val:.4f}')
                self.logging.info(f'R        Eval Loss:{r_loss_val/count_val:.4f}')
                self.logging.info(f'contrast Eval Loss:{contrast_loss_val/count_val:.4f}')
                self.logging.info(f'Avg      Eval Loss:{avg_loss_val:.4f}, Min Loss Epoch: {min_loss_epoch}, Min Loss: {min_loss}\n')

        end_time=time.time()
        print(f'End Training')
        print(f'Training for {end_time-start_time}')
        self.logging.info('End Training')
        print(f'Results save in {self.out_file}')


        t = np.linspace(1,len(train_list),len(train_list))

        ssim_train = [train_list[i][0] for i in range(len(train_list))]
        ssim_val = [val_list[i][0] for i in range(len(val_list))]
        r_train = [train_list[i][1] for i in range(len(train_list))]
        r_val = [val_list[i][1] for i in range(len(val_list))]
        contrast_train = [train_list[i][2] for i in range(len(train_list))]
        contrast_val = [val_list[i][2] for i in range(len(val_list))]
        ## SSIM Loss
        plt.subplot(221)
        plt.plot(t,ssim_train,label='ssim train')
        plt.plot(t,ssim_val,label='ssim val')
        plt.title('SSIM Loss')
        plt.legend()
        ## R Loss
        plt.subplot(222)
        plt.plot(t,r_train,label='r train')
        plt.plot(t,r_val,label='r val')
        plt.title('R Loss')
        plt.legend()
        ## contrast Loss
        plt.subplot(223)
        plt.plot(t,contrast_train,label='contrast train')
        plt.plot(t,contrast_val,label='contrast val')
        plt.title('contrast Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_file,'training_loss.png'))
    
    def start_train(self):
        num_epochs = self.num_epochs
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        self.train(num_epochs,train_loader,eval_loader)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)    

    def data_loader(self,dataset_train_path,dataset_val_path,transform,batch_size,num_workers):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=self.seed_worker, generator=g)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, eval_loader
        
    def save_folder(self):
        out_file = 'exp'
        i = 0
        out_file_tmp = out_file + str(i)
        while(os.path.isdir(out_file_tmp) == True):
            i += 1
            out_file_tmp = out_file + str(i)
        out_file = out_file + str(i)
        os.makedirs(out_file, exist_ok=True)
        print(f'Results save in {out_file}')
        # Copy ace and loss
        shutil.copy('Model.py', out_file)
        shutil.copy('Loss.py', out_file)
        shutil.copy('train.py', out_file)
        return out_file
    
    def logger(self):
        logging.basicConfig(filename=os.path.join(self.out_file,"log.txt"), level=logging.DEBUG, format="%(asctime)s %(message)s")
        return logging

if __name__ == '__main__':
    train = Train()
    train.start_train() 
