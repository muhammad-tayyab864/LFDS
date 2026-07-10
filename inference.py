import os
import cv2
import time
import tqdm
import yaml
import torch
import random
import kornia
import kornia.image
import numpy as np
import matplotlib.pyplot as plt



# ==========================================================
# OLED Power Model (Eq. 5)
# ==========================================================
def oled_power(img_rgb,
               ar=0.265,
               ag=0.670,
               ab=0.065,
               gamma=2.2,
               a0=0):

    img = img_rgb.astype(np.float32) / 255.0

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    power = (
        ar * np.power(R, gamma)
        + ag * np.power(G, gamma)
        + ab * np.power(B, gamma)
    )

    return np.sum(power) + a0


# ==========================================================
# Sensitivity Study Parameters
# ==========================================================
gamma_values = [1.8,1.9,2.0,2.1,2.2,2.3,2.4]

perturb_values = [-0.10,-0.05,0.0,0.05,0.10]

nominal_ar = 0.265
nominal_ag = 0.670
nominal_ab = 0.065






from torchvision import transforms
from torch.utils.data import DataLoader

from Model import Proposed
from variable_luminance import rgb_to_ycbcr, ycbcr_to_rgb
from Dataset_aug import testDataset, kittiTestDataset, middleburyTestDataset, Icvs2017
  
model_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/proposed/exp0'
config_path = os.path.join(model_path,'config.yaml')

pc = 1
dataset_id = 1
dataset_list = ['stereoColorTransferDataset','kitti2015','flickr1024','middleburry2014']
dataset = dataset_list[dataset_id]

rgb = 0
speedup_config = 1

if(os.path.exists(config_path)):

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    hidden_size = config['hidden_size']
    sampling_factor = config['sampling_factor']
    blending_factor = config['blending_factor']

    if(speedup_config == 1):
        speedup = config['speedup']
else:
    speedup = 0
    hidden_size = 16
    sampling_factor = 4
    blending_factor = 0.5

class Inference:
    def __init__(self, resize=None):

        num_workers = 0
        if(resize is None):
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([480,480])])
            
        if(pc == 0):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
            
            elif(dataset == dataset_list[1]):
                dataset_test_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # dataset_test_path1 = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'
                # test_dataset = kittiTestDataset(root=[dataset_test_path0,dataset_test_path1],transform=transform)
            
            elif(dataset == dataset_list[2]):
                dataset_test_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = middleburyTestDataset(root=[dataset_test_path],transform=transform)

        elif(pc == 1):
            if(dataset == dataset_list[0]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/stereoColorTransferDataset/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                
            elif(dataset == dataset_list[1]):
                dataset_test_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
                # dataset_test_path1 = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'
                # test_dataset = kittiTestDataset(root=[dataset_test_path0,dataset_test_path1],transform=transform)
                
            elif(dataset == dataset_list[2]):
                dataset_test_path = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/Flickr1024/Test'
                test_dataset = testDataset(root=[dataset_test_path],transform=transform)
            
            elif(dataset == dataset_list[3]):
                dataset_test_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/middleburry2014/MiddEval3/test/unified'
                test_dataset = middleburyTestDataset(root=[dataset_test_path],transform=transform)
        
        # path
        self.model_path = model_path
        save_path = r'C:/Users/lps3090/Desktop/Testing_StereoOLED/bbiPAM/result'
        self.output_path = os.path.join(save_path,'inference_'+str(dataset))
        os.makedirs(self.output_path,exist_ok=True)
        
        # device
        torch.cuda.get_device_name(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            if(speedup_config == 1):
                self.model = Proposed(hidden_size=hidden_size, sampling_factor=sampling_factor, blending_factor=blending_factor, speedup=speedup, test=1).to(self.device)
            else:
                self.model = Proposed(hidden_size=hidden_size, sampling_factor=sampling_factor, blending_factor=blending_factor, test=1).to(self.device)
        except:
            if(speedup_config == 1):
                self.model = Proposed(hidden_size=hidden_size, sampling_factor=sampling_factor, blending_factor=blending_factor, speedup=speedup).to(self.device)
            else:
                self.model = Proposed(hidden_size=hidden_size, sampling_factor=sampling_factor, blending_factor=blending_factor).to(self.device)
        
        # load model
        self.model.load_state_dict(torch.load(os.path.join(model_path,'best.pth')))

        # load data
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    def decompose_imgs(self, imgs, luminance_const = [0.206, 0.339, 0.454]):
        """
        RGB to YCbCr, returns luminance and chrominance.
        """
        _ycbcr = rgb_to_ycbcr(imgs, luminance_const=luminance_const)
        _y = _ycbcr[:,:1]
        _cbcr = _ycbcr[:,1:]
        return _y, _cbcr
    
    def compose_imgs(self, y_imgs, cbcr_imgs, luminance_const = [0.206, 0.339, 0.454]):
        """
        YCbCr to RGB, returns RGB.
        """
        _ycbcr = torch.cat((y_imgs, cbcr_imgs), dim=1)
        rgb_imgs = ycbcr_to_rgb(_ycbcr, luminance_const=luminance_const)
        return rgb_imgs
        
    def inference(self):        
        self.model.eval()
        # self.model = self.model.half()
        torch.set_grad_enabled(False)

        latency_sum = 0   # latency
        count = 0         # image num


        with torch.no_grad():
            for batch_idx, (inputL, inputR, filenameL, filenameR) in tqdm.tqdm(enumerate(self.test_loader)):

                print(f'\n# =======================================')
                print(f'#         == {filenameL} ==               ')
                print(f'#         == {filenameR} ==               ')
                print(f'# =========================================')

                if(rgb == 1):
                    inputL, inputR = inputL.cuda(), inputR.cuda()

                inputL_y, cbcrL = self.decompose_imgs(inputL)
                inputR_y, cbcrR = self.decompose_imgs(inputR)

                if(rgb == 0):
                    inputL_y, inputR_y = inputL_y.cuda(), inputR_y.cuda()

                R = 0.1     # target R
                R_in = R    # input R
                cnt = 0     # achieved R
                finish = False
                results = {}

                results[0.0] = np.concatenate([inputL, inputR], axis=1)

                while(finish != True):
                    # =================================================
                    #         圖片先將 y 分量提出來輸入至模型訓練 
                    #      輸出之後再將 y, cbcr 組起來，轉換為 rgb
                    # ==================================================
                    if(rgb == 0):  

                        # =========================
                        #       == Timing == 
                        # =========================
                        torch.cuda.synchronize()
                        start_time = time.time()

                        outL_y, outR_y = self.model(inputL_y, inputR_y, R_in)
                        
                        torch.cuda.synchronize()
                        end_time = time.time()

                        # =========================
                        #   == Post-processing == 
                        # =========================
                        outL_y, outR_y = outL_y.cpu(), outR_y.cpu()
                        outL_rgb = torch.clamp(self.compose_imgs(outL_y, cbcrL), 0 ,1)
                        outR_rgb = torch.clamp(self.compose_imgs(outR_y, cbcrR), 0, 1)

                    # =================================================
                    #             圖片直接輸入至模型訓練 
                    #           輸出之後再將 y 分量提出來 
                    # ==================================================
                    else:
                        # =========================
                        #       == Timing == 
                        # =========================
                        torch.cuda.synchronize()
                        start_time = time.time()

                        outL_rgb, outR_rgb = self.model(inputL, inputR, R_in)
                        
                        torch.cuda.synchronize()
                        end_time = time.time()
                        # =========================
                        #   == Post-processing == 
                        # =========================
                        outL_y, _ = self.decompose_imgs(outL_rgb)
                        outR_y, _ = self.decompose_imgs(outR_rgb)

                    # =========================
                    #       == Latency == 
                    # =========================
                    latency = end_time - start_time
                    latency_sum += latency

                    count += 1   # image num
                    cnt += 1     # achieved R

                    # =========================
                    #   == Post-processing == 
                    # =========================
                    outL_rgb = outL_rgb.cpu()
                    outR_rgb = outR_rgb.cpu()
                    
                    outL_rgb = outL_rgb.squeeze().permute(1,2,0)
                    outR_rgb = outR_rgb.squeeze().permute(1,2,0)

                    outL_rgb = outL_rgb.numpy()
                    outR_rgb = outR_rgb.numpy()

                    outL_rgb = np.uint8(outL_rgb*255)
                    outR_rgb = np.uint8(outR_rgb*255)
                    
                    outL_rgb = cv2.cvtColor(outL_rgb, cv2.COLOR_RGB2BGR)
                    outR_rgb = cv2.cvtColor(outR_rgb, cv2.COLOR_RGB2BGR)
                    
                    # print(outL_rgb.shape)
                    img = np.concatenate([outL_rgb,outR_rgb],axis=1)
                    
                    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    #plt.show()

                    # =========================
                    #   == Power Computing == 
                    # =========================
                   # ==========================================================
# INPUT RGB
# ==========================================================

inputL_rgb_np = inputL.squeeze().permute(1,2,0).cpu().numpy()
inputR_rgb_np = inputR.squeeze().permute(1,2,0).cpu().numpy()

inputL_rgb_np = np.uint8(inputL_rgb_np * 255)
inputR_rgb_np = np.uint8(inputR_rgb_np * 255)

# ==========================================================
# OUTPUT RGB
# ==========================================================

outL_rgb_np = cv2.cvtColor(outL_rgb, cv2.COLOR_BGR2RGB)
outR_rgb_np = cv2.cvtColor(outR_rgb, cv2.COLOR_BGR2RGB)

# ==========================================================
# Nominal OLED Model
# ==========================================================

p_in = (
    oled_power(
        inputL_rgb_np,
        nominal_ar,
        nominal_ag,
        nominal_ab,
        gamma=2.2
    )
    +
    oled_power(
        inputR_rgb_np,
        nominal_ar,
        nominal_ag,
        nominal_ab,
        gamma=2.2
    )
)

p_out = (
    oled_power(
        outL_rgb_np,
        nominal_ar,
        nominal_ag,
        nominal_ab,
        gamma=2.2
    )
    +
    oled_power(
        outR_rgb_np,
        nominal_ar,
        nominal_ag,
        nominal_ab,
        gamma=2.2
    )
)

R_actual = 1 - (p_out / p_in)
# ==========================================================
# Gamma Sensitivity
# ==========================================================

for gamma in gamma_values:

    p_in_gamma = (
        oled_power(
            inputL_rgb_np,
            nominal_ar,
            nominal_ag,
            nominal_ab,
            gamma
        )
        +
        oled_power(
            inputR_rgb_np,
            nominal_ar,
            nominal_ag,
            nominal_ab,
            gamma
        )
    )

    p_out_gamma = (
        oled_power(
            outL_rgb_np,
            nominal_ar,
            nominal_ag,
            nominal_ab,
            gamma
        )
        +
        oled_power(
            outR_rgb_np,
            nominal_ar,
            nominal_ag,
            nominal_ab,
            gamma
        )
    )

    saving = 100 * (1 - p_out_gamma / p_in_gamma)

    self.gamma_results[gamma].append(saving)
    
    # ==========================================================
# Red Coefficient Sensitivity
# ==========================================================

for p in perturb_values:

    ar = nominal_ar * (1+p)

    p_in_red = (
        oled_power(
            inputL_rgb_np,
            ar,
            nominal_ag,
            nominal_ab,
            2.2
        )
        +
        oled_power(
            inputR_rgb_np,
            ar,
            nominal_ag,
            nominal_ab,
            2.2
        )
    )

    p_out_red = (
        oled_power(
            outL_rgb_np,
            ar,
            nominal_ag,
            nominal_ab,
            2.2
        )
        +
        oled_power(
            outR_rgb_np,
            ar,
            nominal_ag,
            nominal_ab,
            2.2
        )
    )

    saving = 100 * (1 - p_out_red/p_in_red)

    self.red_results[p].append(saving)
    
    
    # ==========================================================
# Green Coefficient Sensitivity
# ==========================================================

for p in perturb_values:

    ag = nominal_ag * (1+p)

    p_in_green = (
        oled_power(
            inputL_rgb_np,
            nominal_ar,
            ag,
            nominal_ab,
            2.2
        )
        +
        oled_power(
            inputR_rgb_np,
            nominal_ar,
            ag,
            nominal_ab,
            2.2
        )
    )

    p_out_green = (
        oled_power(
            outL_rgb_np,
            nominal_ar,
            ag,
            nominal_ab,
            2.2
        )
        +
        oled_power(
            outR_rgb_np,
            nominal_ar,
            ag,
            nominal_ab,
            2.2
        )
    )

    saving = 100 * (1 - p_out_green/p_in_green)

    self.green_results[p].append(saving)
    
    
    
    # ==========================================================
# Blue Coefficient Sensitivity
# ==========================================================

for p in perturb_values:

    ab = nominal_ab * (1+p)

    p_in_blue = (
        oled_power(
            inputL_rgb_np,
            nominal_ar,
            nominal_ag,
            ab,
            2.2
        )
        +
        oled_power(
            inputR_rgb_np,
            nominal_ar,
            nominal_ag,
            ab,
            2.2
        )
    )

    p_out_blue = (
        oled_power(
            outL_rgb_np,
            nominal_ar,
            nominal_ag,
            ab,
            2.2
        )
        +
        oled_power(
            outR_rgb_np,
            nominal_ar,
            nominal_ag,
            ab,
            2.2
        )
    )

    saving = 100 * (1 - p_out_blue/p_in_blue)

    self.blue_results[p].append(saving)
    
    
    

                    # print(f'R_target = {R:.4f}, R_input = {R_in:.4f}, R_actual = {R_actual:.4f}')

                    # ================================================================================================
                    #                                 模型還沒有達到理想狀態
                    #                   但推理次數過多時，程式仍能儲存當前結果並嘗試新的方向
                    # ================================================================================================
                    if(cnt > 100):
                        # ============================
                        #    == Scaling Factor == 
                        # ============================
                        k = np.power((1 - R), (1/2.2))
                        outL_rgb = inputL.squeeze().permute(1,2,0).cpu().numpy() 
                        outR_rgb = inputR.squeeze().permute(1,2,0).cpu().numpy() 

                        outL_rgb = np.uint8(outL_rgb * 255 * k)
                        outR_rgb = np.uint8(outR_rgb * 255 * k)

                        outL_rgb = cv2.cvtColor(outL_rgb, cv2.COLOR_RGB2BGR)
                        outR_rgb = cv2.cvtColor(outR_rgb, cv2.COLOR_RGB2BGR)
                        
                        cnt = 0
                        print(f'When R = {R:.4f} has Problem')
                        
                        save_path = os.path.join(self.output_path,'R='+str(R))
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path,filenameL[0]), outL_rgb)
                        cv2.imwrite(os.path.join(save_path,filenameR[0]), outR_rgb)

                        if(R > 0.85):
                            finish = True
                            
                        R_in = R + 0.1
                        R = round(R + 0.1, 1)

                    # ================================================================================================
                    #                                           == Ideal R == 
                    # ================================================================================================
                    if(abs(R - R_actual) < 0.005):
                        cnt = 0
                        # print(f'latenct   = {latency:.6f}')
                        # print(f'latenctAvg= {latency_sum/count:.6f}')
                        print('\nIdeal Power-Saving :\n')
                        print(f'        R         = {R:.4f}')
                        print(f'        R_in      = {R_in:.4f}')
                        print(f'        R_actual  = {R_actual:.4f}\n')
                        print(f'====================================================================\n')


                        save_path = os.path.join(self.output_path,'R='+str(R))
                        os.makedirs(save_path, exist_ok = True)
                        cv2.imwrite(os.path.join(save_path,filenameL[0]), outL_rgb)
                        cv2.imwrite(os.path.join(save_path,filenameR[0]), outR_rgb)

                        R_in = R_in * ((R + 0.1) / R)

                        # ============================
                        #   == Complete：R = 0.9 == 
                        # ============================
                        if(R > 0.85):
                            finish = True
                        
                        R = round(R + 0.1, 1)

                    # ================================================================================================
                    #                                      == Not Ideal R == 
                    # ================================================================================================
                    else:
                        # R_in = R_in*abs(float(R-R_actual))
                        # R_in = R_in + float(R-R_actual) * (0.7/(1+cnt*0.2)+1e-5)
                        R_in = R_in + float(R-R_actual) * 0.5
                        # R_in = R_in * (1 + float(R-R_actual))

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


print("\n")
print("================================================")
print("GAMMA SENSITIVITY")
print("================================================")

for gamma in gamma_values:

    vals = np.array(self.gamma_results[gamma])

    mean = np.mean(vals)
    std = np.std(vals)

    print(
        f"Gamma={gamma:.1f} "
        f"Mean={mean:.4f}% "
        f"Std={std:.4f}"
    )

print("\n")
print("================================================")
print("RED COEFFICIENT")
print("================================================")

for p in perturb_values:

    vals = np.array(self.red_results[p])

    print(
        f"{p*100:+.0f}% "
        f"Mean={np.mean(vals):.4f}% "
        f"Std={np.std(vals):.4f}"
    )

print("\n")
print("================================================")
print("GREEN COEFFICIENT")
print("================================================")

for p in perturb_values:

    vals = np.array(self.green_results[p])

    print(
        f"{p*100:+.0f}% "
        f"Mean={np.mean(vals):.4f}% "
        f"Std={np.std(vals):.4f}"
    )

print("\n")
print("================================================")
print("BLUE COEFFICIENT")
print("================================================")

for p in perturb_values:

    vals = np.array(self.blue_results[p])

    print(
        f"{p*100:+.0f}% "
        f"Mean={np.mean(vals):.4f}% "
        f"Std={np.std(vals):.4f}"
    )
