from torch.utils.data.dataset import Dataset
import cv2
import glob
import os
from pathlib import Path

class Compare_flickr1024(Dataset):
    def __init__(self, root1, root2, transform):
        self.transform = transform
        self.data1 = []
        self.data2 = []

        # --- Load data1 ---
        for path in root1:
            file_list_L = sorted(glob.glob(os.path.join(path, "**/*L.*"), recursive=True))
            file_list_R = sorted(glob.glob(os.path.join(path, "**/*R.*"), recursive=True))
            if len(file_list_L) != len(file_list_R):
                print(f"Warning: L and R count mismatch in root1 path {path}")
            self.data1.extend(zip(file_list_L, file_list_R))

        # --- Load data2 ---
        for path in root2:
            file_list_L = sorted(glob.glob(os.path.join(path, "**/*L.*"), recursive=True))
            file_list_R = sorted(glob.glob(os.path.join(path, "**/*R.*"), recursive=True))
            if len(file_list_L) != len(file_list_R):
                print(f"Warning: L and R count mismatch in root2 path {path}")
            self.data2.extend(zip(file_list_L, file_list_R))

        # Ensure matching length to prevent index errors
        min_len = min(len(self.data1), len(self.data2))
        self.data1 = self.data1[:min_len]
        self.data2 = self.data2[:min_len]

        print(f"Loaded {len(self.data1)} image pairs for data1 and data2.")

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        imgL_path1, imgR_path1 = self.data1[idx]
        imgL_path2, imgR_path2 = self.data2[idx]

        def load_and_transform(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            k = 8
            img = cv2.resize(img, ((w // k) * k, (h // k) * k))
            return self.transform(img)

        img1L_tensor = load_and_transform(imgL_path1)
        img1R_tensor = load_and_transform(imgR_path1)
        img2L_tensor = load_and_transform(imgL_path2)
        img2R_tensor = load_and_transform(imgR_path2)

        filenameL = os.path.basename(imgL_path1) + '/' + os.path.basename(imgL_path2)
        filenameR = os.path.basename(imgR_path1) + '/' + os.path.basename(imgR_path2)

        return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR


# class Compare_flickr1024(Dataset):
#     def __init__(self,root1,root2,transform):
#         self.transform = transform
        
#         self.imgs_path1 = root1
#         self.data1 = []
#         self.data_list1_L = []
#         self.data_list1_R = []

#         for i in range(len(self.imgs_path1)):
#             file_list1_L = glob.glob(self.imgs_path1[i] + "*/*L.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "*/*L.png",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*L.bmp",recursive=True)
#             file_list1_R = glob.glob(self.imgs_path1[i] + "*/*R.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "*/*R.png",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*R.bmp",recursive=True)
#             for j in range(len(file_list1_L)):
#                 self.data_list1_L.append(file_list1_L[j])
#             for j in range(len(file_list1_R)):
#                 self.data_list1_R.append(file_list1_R[j])

#             for i in range(len(self.data_list1_L)):
#                 self.data1.append([self.data_list1_L[i],self.data_list1_R[i]])

#         self.imgs_path2 = root2
#         self.data2 = []
#         self.data_list2_L = []
#         self.data_list2_R = []

#         for i in range(len(self.imgs_path2)):
#             file_list2_L = glob.glob(self.imgs_path2[i] + "*/*L.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "*/*L.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*L.bmp",recursive=True)
#             file_list2_R = glob.glob(self.imgs_path2[i] + "*/*R.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "*/*R.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*R.bmp",recursive=True)
#             for j in range(len(file_list2_L)):
#                 self.data_list2_L.append(file_list2_L[j])
#             for j in range(len(file_list2_R)):
#                 self.data_list2_R.append(file_list2_R[j])

#             for i in range(len(self.data_list2_L)):
#                 self.data2.append([self.data_list2_L[i],self.data_list2_R[i]])
        
#         print(len(self.data1))
#         print(len(self.data2))

#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
        
#         imgL_path1 = self.data1[idx][0]
#         imgR_path1 = self.data1[idx][1]

#         img1L = cv2.imread(imgL_path1)
#         img1R = cv2.imread(imgR_path1)
#         img1L = cv2.cvtColor(img1L,cv2.COLOR_BGR2RGB)
#         img1R = cv2.cvtColor(img1R,cv2.COLOR_BGR2RGB)

#         h,w = img1L.shape[:2]
#         k=8
#         img1L = cv2.resize(img1L,((w//k)*k,(h//k)*k))
#         img1R = cv2.resize(img1R,((w//k)*k,(h//k)*k))

#         img1L_tensor = self.transform(img1L)
#         img1R_tensor = self.transform(img1R)

#         imgL_path2 = self.data2[idx][0]
#         imgR_path2 = self.data2[idx][1]

#         img2L = cv2.imread(imgL_path2)
#         img2R = cv2.imread(imgR_path2)
#         img2L = cv2.cvtColor(img2L,cv2.COLOR_BGR2RGB)
#         img2R = cv2.cvtColor(img2R,cv2.COLOR_BGR2RGB)

#         h,w = img2L.shape[:2]
#         img2L = cv2.resize(img2L,((w//k)*k,(h//k)*k))
#         img2R = cv2.resize(img2R,((w//k)*k,(h//k)*k))

#         img2L_tensor = self.transform(img2L)
#         img2R_tensor = self.transform(img2R)

#         filenameL = os.path.split(imgL_path1)[-1] + '/' + os.path.split(imgL_path2)[-1]
#         filenameR = os.path.split(imgR_path1)[-1] + '/' + os.path.split(imgR_path2)[-1]

#         return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR


class StereoColorTransfer(Dataset):
    def __init__(self, root1, root2, transform):
        self.transform = transform
        self.data1 = []
        self.data2 = []

        # --- Load reference images (data1) ---
        for path in root1:
            file_list_L = sorted(glob.glob(os.path.join(path, "**/*L.*"), recursive=True))
            file_list_R = sorted(glob.glob(os.path.join(path, "**/*R.*"), recursive=True))
            if len(file_list_L) != len(file_list_R):
                print(f"⚠️ Warning: L and R count mismatch in root1 path {path}")
            self.data1.extend(zip(file_list_L, file_list_R))

        # --- Load color-transferred images (data2) ---
        for path in root2:
            file_list_L = sorted(glob.glob(os.path.join(path, "**/*L.*"), recursive=True))
            file_list_R = sorted(glob.glob(os.path.join(path, "**/*R.*"), recursive=True))
            if len(file_list_L) != len(file_list_R):
                print(f"⚠️ Warning: L and R count mismatch in root2 path {path}")
            self.data2.extend(zip(file_list_L, file_list_R))

        # --- Ensure lengths match to avoid index errors ---
        min_len = min(len(self.data1), len(self.data2))
        self.data1 = self.data1[:min_len]
        self.data2 = self.data2[:min_len]

        print(f"✅ Loaded {len(self.data1)} stereo image pairs for comparison.")

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        imgL_path1, imgR_path1 = self.data1[idx]
        imgL_path2, imgR_path2 = self.data2[idx]

        def load_and_transform(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            k = 8  # Ensure dimensions divisible by 8
            img = cv2.resize(img, ((w // k) * k, (h // k) * k))
            return self.transform(img)

        # Apply transform to each image
        refL = load_and_transform(imgL_path1)
        refR = load_and_transform(imgR_path1)
        outL = load_and_transform(imgL_path2)
        outR = load_and_transform(imgR_path2)

        # Generate filenames for traceability
        filenameL = os.path.basename(imgL_path1) + '/' + os.path.basename(imgL_path2)
        filenameR = os.path.basename(imgR_path1) + '/' + os.path.basename(imgR_path2)

        return (refL, refR), (outL, outR), filenameL, filenameR



class Compare_kitti2015(Dataset):
    def __init__(self, root1, root2, transform):
        self.transform = transform

        # Load data1 (primary image pair set)
        self.imgs_path1 = root1
        file_list1_L = sorted(
            glob.glob(self.imgs_path1[0] + "*/*.jpg", recursive=True)
            + glob.glob(self.imgs_path1[0] + "*/*.png", recursive=True)
            + glob.glob(self.imgs_path1[0] + "/**/*.bmp", recursive=True)
        )
        file_list1_R = sorted(
            glob.glob(self.imgs_path1[1] + "*/*.jpg", recursive=True)
            + glob.glob(self.imgs_path1[1] + "*/*.png", recursive=True)
            + glob.glob(self.imgs_path1[1] + "/**/*.bmp", recursive=True)
        )

        # Pairing data1
        self.data1 = list(zip(file_list1_L, file_list1_R))

        # Load data2 (comparison image pair set)
        self.imgs_path2 = root2
        self.data2 = []

        for folder in self.imgs_path2:
            file_list2_L = sorted(
    
                glob.glob(folder + "*/*_2.jpg", recursive=True)
                + glob.glob(folder + "*/*_2.png", recursive=True)
                + glob.glob(folder + "/**/*_2.bmp", recursive=True)
            )
            file_list2_R = sorted(
                glob.glob(folder + "*/*_3.jpg", recursive=True)
                + glob.glob(folder + "*/*_3.png", recursive=True)
                + glob.glob(folder + "/**/*_3.bmp", recursive=True)
            )
            self.data2 += list(zip(file_list2_L, file_list2_R))

        # Sanity check
        print(f"[INFO] Loaded {len(self.data1)} pairs in data1")
        print(f"[INFO] Loaded {len(self.data2)} pairs in data2")

        # Set valid dataset size to prevent indexing errors
        self.dataset_length = min(len(self.data1), len(self.data2))
        if self.dataset_length == 0:
            print("[WARNING] One or both datasets are empty. Dataset will return no samples.")

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        imgL_path1, imgR_path1 = self.data1[idx]
        imgL_path2, imgR_path2 = self.data2[idx]

        # Load and preprocess images from data1
        img1L = cv2.cvtColor(cv2.imread(imgL_path1), cv2.COLOR_BGR2RGB)
        img1R = cv2.cvtColor(cv2.imread(imgR_path1), cv2.COLOR_BGR2RGB)

        # Resize to nearest multiple of 8
        h, w = img1L.shape[:2]
        k = 8
        img1L = cv2.resize(img1L, ((w // k) * k, (h // k) * k))
        img1R = cv2.resize(img1R, ((w // k) * k, (h // k) * k))

        img1L_tensor = self.transform(img1L)
        img1R_tensor = self.transform(img1R)

        # Load and preprocess images from data2
        img2L = cv2.cvtColor(cv2.imread(imgL_path2), cv2.COLOR_BGR2RGB)
        img2R = cv2.cvtColor(cv2.imread(imgR_path2), cv2.COLOR_BGR2RGB)

        h, w = img2L.shape[:2]
        img2L = cv2.resize(img2L, ((w // k) * k, (h // k) * k))
        img2R = cv2.resize(img2R, ((w // k) * k, (h // k) * k))

        img2L_tensor = self.transform(img2L)
        img2R_tensor = self.transform(img2R)

        filenameL = os.path.basename(imgL_path1) + '/' + os.path.basename(imgL_path2)
        filenameR = os.path.basename(imgR_path1) + '/' + os.path.basename(imgR_path2)

        return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR





# class Compare_kitti2015(Dataset):
#     def __init__(self,root1,root2,transform):
#         self.transform = transform

#         self.imgs_path1 = root1
#         self.data1 = []
        
#         self.data_list1_L = []
#         self.data_list1_R = []
#         file_list1_L = glob.glob(self.imgs_path1[0] + "*/*.jpg",recursive=True) \
#                      + glob.glob(self.imgs_path1[0] + "*/*.png",recursive=True) \
#                      + glob.glob(self.imgs_path1[0] + "/**/*.bmp",recursive=True)
#         file_list1_R = glob.glob(self.imgs_path1[1] + "*/*.jpg",recursive=True) \
#                      + glob.glob(self.imgs_path1[1] + "*/*.png",recursive=True) \
#                      + glob.glob(self.imgs_path1[1] + "/**/*.bmp",recursive=True)

#         for i in range(len(file_list1_L)):
#             self.data1.append([file_list1_L[i],file_list1_R[i]])

#         self.imgs_path2 = root2
#         self.data2 = []
        
#         for i in range(len(self.imgs_path2)):
#             self.data_list2_L = []
#             self.data_list2_R = []
#             file_list2_L = glob.glob(self.imgs_path2[i] + "*/*_2.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "*/*_2.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*_2.bmp",recursive=True)
#             file_list2_R = glob.glob(self.imgs_path2[i] + "*/*_3.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "*/*_3.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*_3.bmp",recursive=True)
#             for j in range(len(file_list2_L)):
#                 self.data_list2_L.append(file_list2_L[j])
#             for j in range(len(file_list2_R)):
#                 self.data_list2_R.append(file_list2_R[j])

#             for i in range(len(self.data_list2_L)):
#                 self.data2.append([self.data_list2_L[i],self.data_list2_R[i]])


#         print(len(self.data1))
#         print(len(self.data2))


#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
#         imgL_path1 = self.data1[idx][0]
#         imgR_path1 = self.data1[idx][1]

#         img1L = cv2.imread(imgL_path1)
#         img1R = cv2.imread(imgR_path1)
#         img1L = cv2.cvtColor(img1L,cv2.COLOR_BGR2RGB)
#         img1R = cv2.cvtColor(img1R,cv2.COLOR_BGR2RGB)

#         h,w = img1L.shape[:2]
#         k=8
#         img1L = cv2.resize(img1L,((w//k)*k,(h//k)*k))
#         img1R = cv2.resize(img1R,((w//k)*k,(h//k)*k))

#         img1L_tensor = self.transform(img1L)
#         img1R_tensor = self.transform(img1R)


#         imgL_path2 = self.data2[idx][0]
#         imgR_path2 = self.data2[idx][1]

    

#         img2L = cv2.imread(imgL_path2)
#         img2R = cv2.imread(imgR_path2)
#         img2L = cv2.cvtColor(img2L,cv2.COLOR_BGR2RGB)
#         img2R = cv2.cvtColor(img2R,cv2.COLOR_BGR2RGB)

#         h,w = img2L.shape[:2]
#         img2L = cv2.resize(img2L,((w//k)*k,(h//k)*k))
#         img2R = cv2.resize(img2R,((w//k)*k,(h//k)*k))

#         img2L_tensor = self.transform(img2L)
#         img2R_tensor = self.transform(img2R)

#         filenameL = os.path.split(imgL_path1)[-1] + '/' + os.path.split(imgL_path2)[-1]
#         filenameR = os.path.split(imgR_path1)[-1] + '/' + os.path.split(imgR_path2)[-1]



#         return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR
    

class Compare_middlebury2006(Dataset):
    def __init__(self, root1, root2, transform):
        self.transform = transform
        self.data1 = []
        self.data2 = []

        def load_pairs(root_paths, view1_suffix="view1", view5_suffix="view5"):
            data = []
            for path in root_paths:
                # Load left and right images for each path
                left_images = glob.glob(os.path.join(path, "**", f"*{view1_suffix}.jpg"), recursive=True)
                left_images += glob.glob(os.path.join(path, "**", f"*{view1_suffix}.png"), recursive=True)
                left_images += glob.glob(os.path.join(path, "**", f"*{view1_suffix}.bmp"), recursive=True)

                right_images = glob.glob(os.path.join(path, "**", f"*{view5_suffix}.jpg"), recursive=True)
                right_images += glob.glob(os.path.join(path, "**", f"*{view5_suffix}.png"), recursive=True)
                right_images += glob.glob(os.path.join(path, "**", f"*{view5_suffix}.bmp"), recursive=True)

                # Sort to ensure matching order
                left_images.sort()
                right_images.sort()

                # Match only min pairs
                min_len = min(len(left_images), len(right_images))
                data.extend(zip(left_images[:min_len], right_images[:min_len]))
            return data

        self.data1 = load_pairs(root1)
        self.data2 = load_pairs(root2)

        min_len = min(len(self.data1), len(self.data2))
        self.data1 = self.data1[:min_len]
        self.data2 = self.data2[:min_len]

        print(f"Loaded {len(self.data1)} image pairs for Middlebury 2006")

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        def load_and_preprocess(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            k = 8
            img = cv2.resize(img, ((w // k) * k, (h // k) * k))
            return self.transform(img)

        imgL_path1, imgR_path1 = self.data1[idx]
        imgL_path2, imgR_path2 = self.data2[idx]

        img1L_tensor = load_and_preprocess(imgL_path1)
        img1R_tensor = load_and_preprocess(imgR_path1)
        img2L_tensor = load_and_preprocess(imgL_path2)
        img2R_tensor = load_and_preprocess(imgR_path2)

        filenameL = os.path.basename(imgL_path1) + '/' + os.path.basename(imgL_path2)
        filenameR = os.path.basename(imgR_path1) + '/' + os.path.basename(imgR_path2)

        return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR



# class Compare_middlebury2006(Dataset):
#     def __init__(self,root1,root2,transform):
#         self.transform = transform

#         self.imgs_path1 = root1
#         self.data1 = []
        
#         for i in range(len(self.imgs_path1)):
#             self.data_list1_L = []
#             self.data_list1_R = []
#             file_list1_L = glob.glob(self.imgs_path1[i] + "/**/*view1.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*view1.png",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*view1.bmp",recursive=True)
#             file_list1_R = glob.glob(self.imgs_path1[i] + "/**/*view5.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*view5.png",recursive=True) \
#                         + glob.glob(self.imgs_path1[i] + "/**/*view5.bmp",recursive=True)
#             for j in range(len(file_list1_L)):
#                 self.data_list1_L.append(file_list1_L[j])
#             for j in range(len(file_list1_R)):
#                 self.data_list1_R.append(file_list1_R[j])

#             for i in range(len(self.data_list1_L)):
#                 self.data1.append([self.data_list1_L[i],self.data_list1_R[i]])

#         self.imgs_path2 = root2
#         self.data2 = []
        
#         for i in range(len(self.imgs_path2)):
#             self.data_list2_L = []
#             self.data_list2_R = []
#             file_list2_L = glob.glob(self.imgs_path2[i] + "/**/*view1.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*view1.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*view1.bmp",recursive=True)
#             file_list2_R = glob.glob(self.imgs_path2[i] + "/**/*view5.jpg",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*view5.png",recursive=True) \
#                         + glob.glob(self.imgs_path2[i] + "/**/*view5.bmp",recursive=True)
#             for j in range(len(file_list2_L)):
#                 self.data_list2_L.append(file_list2_L[j])
#             for j in range(len(file_list2_R)):
#                 self.data_list2_R.append(file_list2_R[j])

#             for i in range(len(self.data_list2_L)):
#                 self.data2.append([self.data_list2_L[i],self.data_list2_R[i]])

#         print(len(self.data1))
#         print(len(self.data2))


#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
#         imgL_path1 = self.data1[idx][0]
#         imgR_path1 = self.data1[idx][1]

#         img1L = cv2.imread(imgL_path1)
#         img1R = cv2.imread(imgR_path1)
#         img1L = cv2.cvtColor(img1L,cv2.COLOR_BGR2RGB)
#         img1R = cv2.cvtColor(img1R,cv2.COLOR_BGR2RGB)

#         h,w = img1L.shape[:2]
#         k=8
#         img1L = cv2.resize(img1L,((w//k)*k,(h//k)*k))
#         img1R = cv2.resize(img1R,((w//k)*k,(h//k)*k))

#         img1L_tensor = self.transform(img1L)
#         img1R_tensor = self.transform(img1R)

#         imgL_path2 = self.data2[idx][0]
#         imgR_path2 = self.data2[idx][1]

#         img2L = cv2.imread(imgL_path2)
#         img2R = cv2.imread(imgR_path2)
#         img2L = cv2.cvtColor(img2L,cv2.COLOR_BGR2RGB)
#         img2R = cv2.cvtColor(img2R,cv2.COLOR_BGR2RGB)

#         h,w = img2L.shape[:2]
#         img2L = cv2.resize(img2L,((w//k)*k,(h//k)*k))
#         img2R = cv2.resize(img2R,((w//k)*k,(h//k)*k))

#         img2L_tensor = self.transform(img2L)
#         img2R_tensor = self.transform(img2R)

#         filenameL = os.path.split(imgL_path1)[-1] + '/' + os.path.split(imgL_path2)[-1]
#         filenameR = os.path.split(imgR_path1)[-1] + '/' + os.path.split(imgR_path2)[-1]

#         return (img1L_tensor, img1R_tensor), (img2L_tensor, img2R_tensor), filenameL, filenameR
