#################################StereoOLED######################################

# import os
# import shutil

# # Define your directories
# left_dir = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/left/testing'        # Replace with actual path
# right_dir = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/right/testing'      # Replace with actual path
# combined_dir = 'C:/Users/lps3090/Desktop/Testing_StereoOLED/Datasets/KITTI/test'  # Output directory

# # Create output directory if it doesn't exist
# os.makedirs(combined_dir, exist_ok=True)

# # Function to copy and rename images
# def process_images(source_dir, postfix, start_index=1):
#     image_files = sorted(os.listdir(source_dir))  # Sort to maintain order
#     for idx, filename in enumerate(image_files, start=start_index):
#         name, ext = os.path.splitext(filename)
#         new_name = f"{idx:04d}_{postfix}{ext}"
#         src_path = os.path.join(source_dir, filename)
#         dst_path = os.path.join(combined_dir, new_name)
#         shutil.copy2(src_path, dst_path)

# # Process both left and right directories
# process_images(left_dir, "L")
# process_images(right_dir, "R")

# print("Images have been renamed and copied to the combined directory.")

#################################Low-light######################################
# import os

# def rename_images(folder_path):
#     # List all files in the folder
#     files = os.listdir(folder_path)
    
#     # Filter only image files (common extensions)
#     image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
#     images = [f for f in files if f.lower().endswith(image_extensions)]
    
#     # Sort to maintain order (optional)
#     images.sort()
    
#     # Rename images sequentially
#     for i, filename in enumerate(images, start=1):
#         ext = os.path.splitext(filename)[1]  # Keep original extension
#         new_name = f"{i}{ext}"
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_name)
        
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} → {new_name}")

# # Example usage:
# folder = r"C:/Users/lps3090/Desktop/Low-light/MECR-LLE/CUE/data/FiveK/test/groundtruth"  # Change to your folder path
# rename_images(folder)

################################Low-light######################################
import os

def rename_jpg_files(directory):
    """
    Renames all files ending with '.JPG' to '.jpg' in the given directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            old_path = os.path.join(directory, filename)
            new_filename = filename[:-4] + ".jpg"  # Replace .JPG with .jpg
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage:
dir1 = "C:/Users/lps3090/Desktop/Low-light/MECR-LLE/CUE/data/FiveK/test/groundtruth"
dir2 = "C:/Users/lps3090/Desktop/Low-light/MECR-LLE/CUE/data/FiveK/test/input"

rename_jpg_files(dir1)
rename_jpg_files(dir2)
