import torch
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Add the path to the medical_data directory
medical_data_path = os.path.abspath("/home/heming/research/medical/ControlNetPlus/medical_data")
if medical_data_path not in sys.path:
    sys.path.append(medical_data_path)
from common import transformData, dataIO
import glob

io=dataIO() 
transform = transformData()

class Train_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET"], patch_size=128, medical_embeddings_path="/home/heming/research/medical/ControlNetPlus/medical_data/modality_embeddings.pt"):
        self.LQ_paths = [] 
        self.HQ_paths = [] 
        
        for modality in modality_list:
            
            tmp_paths = glob.glob(os.path.join(root_dir, modality, "train", "LQ", "*.bin")) 
            
            for p in tmp_paths:  
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))

        self.length = len(self.LQ_paths) 
        # self.label_dict = {
        #     "PET": 0, 
        #     "CT": 1, 
        #     "MRI": 2
        #     } 
        self.label_dict = {
            "PET": "PET", 
            "CT": "CT", 
            "MRI": "MRI"
            } 
        self.patch_size = patch_size
        
        
        # Add column names to mimic Hugging Face dataset
        '''
        image -> HQ image
        caption -> modality
        conditioning_image -> LQ image
        '''
        self.column_names = ["image", "caption", "conditioning_image"] #, "pixel_values", "conditioning_pixel_values"]
        self.modality_embeddings = torch.load(medical_embeddings_path)
        # print(self.modality_embeddings)
        
        # Preprocessing transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )



    def __len__(self):
        return self.length 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx]) 
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace() 
        modality, _ = self.analyze_path(self.LQ_paths[idx]) 
        
        imgLQ = transform.normalize(imgLQ, modality)
        imgHQ = transform.normalize(imgHQ, modality)

        cat_pic = torch.cat([imgLQ, imgHQ], dim=0).unsqueeze(1)
        cat_pic = transform.random_crop(tensor = cat_pic, patch_size=[self.patch_size, self.patch_size]).squeeze(1)
        imgLQ, imgHQ = torch.chunk(cat_pic, 2, dim=0) 
        
        # Apply preprocessing transforms
        # maybe need image.convert("RGB"), but maybe not, as all those medical images are grayscale images
        # still need as the input need to have 3 channel
        # Convert tensors to NumPy arrays
        # Convert tensors to PIL images
        imgLQ = to_pil_image(imgLQ.squeeze(0)).convert("RGB")  # Convert LQ image
        imgHQ = to_pil_image(imgHQ.squeeze(0)).convert("RGB")  # Convert HQ image
        
        if self.conditioning_image_transforms:
            conditioning_pixel_values = self.conditioning_image_transforms(imgLQ)  # Conditioning image
        if self.image_transforms:
            pixel_values = self.image_transforms(imgHQ)  # Main image
        
        class_label = self.label_dict[modality]
        prompt_embeds = self.modality_embeddings[modality]["prompt_embeds"]
        text_embeds = self.modality_embeddings[modality]["text_embeds"]
        time_ids = self.modality_embeddings[modality]["time_ids"]
        # text_embeds_reduced = text_embeds[:, :time_ids.size(-1)]
        # print(f'text_embeds: {text_embeds}')
        # print(f'time_embeds: {text_embeds}')
        
        item_dict = {"caption": class_label, "pixel_values": pixel_values, "conditioning_pixel_values": conditioning_pixel_values, "prompt_embeds": prompt_embeds, "unet_added_conditions": {"text_embeds": text_embeds, "time_ids": time_ids}}
        # print('item_dict', item_dict)
        # return imgLQ, imgHQ, class_label
        # TODO: maybe can change class_label to certain text prompt
        return item_dict


class Test_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET", "CT", "MRI"], use_num = None, target_folder="validation"): 
        
        self.LQ_paths = [] 
        self.HQ_paths = [] 
        
        
        
        for modality in modality_list: 
            tmp_paths = glob.glob(os.path.join(root_dir, modality, target_folder, "LQ", "*.nii")) 
            
            use_num = len(tmp_paths) if use_num is None else use_num
            
            for num in range(use_num): 
                p = tmp_paths[num]
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))  

        self.length = len(self.LQ_paths) 

    def analyze_path(self, path): 
        path_parts = path.split('/') 
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name
        

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx])
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        modality, file_name = self.analyze_path(self.LQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace()

        
        imgLQ = transform.normalize(imgLQ, modality) 
        imgHQ = transform.normalize(imgHQ, modality) 
        
        imgLQ = torch.from_numpy(imgLQ).unsqueeze(0) 
        imgHQ = torch.from_numpy(imgHQ).unsqueeze(0)

        return imgLQ, imgHQ, modality, file_name








class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.len = len(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except Exception as e:
            # 如果 DataLoader 中的数据采样完了，重新 shuffle 数据
            # print(f"Exception occurred: {e}")
            # exit()
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
            

        return batch

# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 

if __name__ == "__main__": 
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    
    data_root = "/home/heming/dataset/medical/All-in-One" 
    modality_list = ["PET", "CT", "MRI"]
    train_loader_list = []


    
    dataset = {
        'train': Train_Data(root_dir = data_root, modality = "PET"), 
        # 'val': Test_Data(root_dir = data_root, modality_list = ["MRI"], use_num=4), 
        # 'test': Test_Data(root_dir=data_root, center_name="m660-1", use_num=-1),
        
        } 
    train_loader = DataLoader(dataset['train'], batch_size=1, shuffle=False) 
    
    print("length:", len(train_loader))
    
    # valid_loader = DataLoader(Val_Data(root=data_root, center_list=center_list), batch_size=1) 
    
    # test_loader = DataLoader(Test_Data(root=data_root, center_list=center_list), batch_size=1)
    
    

    for counter, data in enumerate(tqdm(train_loader)): 
        import pdb 
        pdb.set_trace()